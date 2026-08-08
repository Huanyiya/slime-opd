import ray
import time

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils import logging_utils
from slime.utils.logging_utils import configure_logger, finish_tracking, init_tracking
from slime.utils.metric_utils import (
    add_train_step_metric,
    compute_first_train_step_for_rollout,
    compute_train_step_before_rollout,
    compute_train_steps_per_rollout,
)
from slime.utils.misc import should_run_periodic_action


def _split_rollout_result(args, rollout_result):
    if args.log_opd_phase_times_only:
        rollout_data_ref, rollout_phase_metrics = rollout_result
        return rollout_data_ref, rollout_phase_metrics or {}
    return rollout_result, {}


def _collect_actor_phase_metrics(actor_train_results):
    for item in actor_train_results or []:
        if isinstance(item, dict) and item:
            return item
    return {}


def _log_opd_phase_times(
    args,
    metric_step,
    rollout_phase_metrics,
    actor_phase_metrics,
    update_phase_metrics,
    step_start,
    excluded_time=0.0,
):
    if not args.log_opd_phase_times_only:
        return

    log_dict = {
        "perf/student_rollout_time": float(rollout_phase_metrics.get("student_rollout_time", 0.0)),
        "perf/teacher_score_time": float(rollout_phase_metrics.get("teacher_score_time", 0.0)),
        "perf/student_logprob_forward_time": float(actor_phase_metrics.get("student_logprob_forward_time", 0.0)),
        "perf/student_train_time": float(actor_phase_metrics.get("student_train_time", 0.0)),
        "perf/weight_sync_time": float(update_phase_metrics.get("weight_sync_time", 0.0)),
        "perf/step_total_time": max(time.time() - step_start - excluded_time, 0.0),
    }
    add_train_step_metric(log_dict, metric_step)
    logging_utils.log(args, log_dict, step_key="step")


def _crossed_eval_interval(previous_step: int, current_step: int, interval: int | None) -> bool:
    if interval is None or current_step <= previous_step:
        return False
    return current_step // interval > previous_step // interval


def train(args):
    configure_logger()
    release_train = args.release_train

    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    # Always push actor weights to rollout once weights are loaded.
    split_disk_sync = actor_model.uses_split_disk_weight_sync()
    if split_disk_sync:
        initial_export = actor_model.export_weights_to_disk()
        if release_train:
            actor_model.release()
        else:
            actor_model.offload()
        actor_model.reload_rollout_weights_from_disk(initial_export)
    else:
        if args.offload_rollout and not release_train:
            ray.get(rollout_manager.onload_weights.remote())
        actor_model.update_weights()

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())

    steps_per_rollout = compute_train_steps_per_rollout(args)
    current_train_step = compute_train_step_before_rollout(args, args.start_rollout_id)

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0, metric_step=0))

    def offload_train(actor_trains_this_step):
        # Each model auto-offloads after train() when offload_train is set,
        # so we only need clear_memory for the non-offload case.
        if not args.offload_train:
            if not args.use_critic or actor_trains_this_step:
                actor_model.clear_memory()
            else:
                critic_model.clear_memory()

    # train loop.
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            ray.get(rollout_manager.eval.remote(rollout_id, metric_step=current_train_step))

        step_start = time.time()
        rollout_metric_step = compute_first_train_step_for_rollout(args, rollout_id)
        rollout_result = ray.get(rollout_manager.generate.remote(rollout_id, metric_step=rollout_metric_step))
        rollout_data_ref, rollout_phase_metrics = _split_rollout_result(args, rollout_result)

        if args.offload_rollout:
            ray.get(rollout_manager.offload.remote())

        if release_train:
            actor_model.create()

        actor_trains = (not args.use_critic) or rollout_id >= args.num_critic_only_steps
        if args.use_critic:
            value_refs = critic_model.async_train(rollout_id, rollout_data_ref)
            if actor_trains:
                actor_train_results = ray.get(actor_model.async_train(rollout_id, rollout_data_ref, external_data=value_refs))
            else:
                ray.get(value_refs)
                actor_train_results = []
        else:
            actor_train_results = ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
        actor_phase_metrics = _collect_actor_phase_metrics(actor_train_results)
        previous_train_step = current_train_step
        current_train_step += steps_per_rollout

        excluded_time = 0.0
        if release_train or should_run_periodic_action(
            rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout
        ):
            save_start = time.time()
            force_sync = release_train or rollout_id == args.num_rollout - 1
            if actor_trains:
                actor_model.save_model(rollout_id, force_sync=force_sync)
            if args.use_critic:
                critic_model.save_model(rollout_id, force_sync=force_sync)
            if args.rollout_global_dataset:
                ray.get(rollout_manager.save.remote(rollout_id))
            excluded_time += time.time() - save_start

        if actor_trains and split_disk_sync:
            export_result = actor_model.export_weights_to_disk()
            if release_train:
                actor_model.release()
            else:
                actor_model.offload()
            update_phase_metrics = actor_model.reload_rollout_weights_from_disk(export_result)
        else:
            offload_train(actor_trains)
            if args.offload_rollout and not release_train:
                ray.get(rollout_manager.onload_weights.remote())
            update_phase_metrics = actor_model.update_weights() if actor_trains else {}
        _log_opd_phase_times(
            args,
            current_train_step,
            rollout_phase_metrics,
            actor_phase_metrics,
            update_phase_metrics or {},
            step_start,
            excluded_time=excluded_time,
        )

        if args.offload_rollout:
            ray.get(rollout_manager.onload_kv.remote())

        if _crossed_eval_interval(previous_train_step, current_train_step, args.eval_interval):
            ray.get(rollout_manager.eval.remote(rollout_id, metric_step=current_train_step))

    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


if __name__ == "__main__":
    args = parse_args()
    train(args)
