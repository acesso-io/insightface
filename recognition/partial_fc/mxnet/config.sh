export CUDA_VISIBLE_DEVICES='0'
export HOROVOD_GPU_ALLREDUCE=NCCL
export HOROVOD_GPU_ALLGATHER=NCCL
export HOROVOD_GPU_BROADCAST=NCLL
export MXNET_CPU_WORKER_NTHREADS=3

# use `which python` to get the absolute path of your python interpreter
#
PYTHON_EXEC=/usr/bin/python
${PYTHON_EXEC} train_memory.py \
--dataset emore \
--loss cosface \
--network r100