from model.VSL_Base import VSLBase
import options
import submitit

def create_executor(configs):
    executor = submitit.AutoExecutor(folder=configs.slurm_log_folder)
    executor.update_parameters(
        timeout_min=configs.slurm_timeout_min,
        constraint=configs.slurm_constraint,
        slurm_partition=configs.slurm_partition,
        gpus_per_node=configs.slurm_gpus,
        cpus_per_task=configs.slurm_cpus,
    )
    return executor

if __name__ == "__main__":
    configs, parser = options.read_command_line()
    if not configs.slurm:
        VSLBase(configs).run()
    else:
        executor = create_executor(configs)
        job = executor.submit(VSLBase(configs).run)
        print("job=", job.job_id)
        if configs.slurm_wait:
            job.result()
