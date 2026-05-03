'''Data-parallelism strategy dialects.

Each strategy decides how `ParallelFor` distributes work across
threads/warps/blocks. A pipeline picks one at lowering time
based on its workload characteristics (uniform vs skewed,
fixed-shape vs dynamic).
'''
