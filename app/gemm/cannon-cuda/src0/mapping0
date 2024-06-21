Task * GPU,OMP,CPU;
# Task taco_validate CPU;
# region $taskname $region_name $processor $list_of_memories;
Region * * * SOCKMEM,SYSMEM;
Region * * GPU FBMEM,ZCMEM;

# Other supported Memory: RDMEM

# Layout taskname regionname memory AOS F_order;
Layout * * * SOA C_order; # Align==128 Compact

InstanceLimit task_4 1; # controlled by command line in TacoMapper
InstanceLimit task_6 1; # controlled by command line in TacoMapper

CollectMemory task_4 *; # controlled by command line in TacoMapper

def block_primitive(IPoint x, ISpace y, MSpace z, int dim1, int dim2) {
    return x[dim1] * z.size[dim2] / y.size[dim1];
}

def cyclic_primitive(IPoint x, ISpace y, MSpace z, int dim1, int dim2) {
    return x[dim1] % z.size[dim2];
}

m_2d = Machine(GPU); # nodes * processors
m_1d = m_2d.merge(0, 1);

def auto2d(Task task) {
    m_3d = m_2d.auto_split(0, task.ispace); # split the original 0 dim into 0,1 dim
    # subspace: task.ispace / m_3d[:-1]
    # split the processor (previosly 1, now 2) dim into 2,3 dim w.r.t subspace
    m_4d = m_3d.auto_split(2, task.ispace / m_3d[:-1]);
    upper = tuple(block_primitive(task.ipoint, task.ispace, m_4d, i, i) for i in (0,1));
    lower = tuple(cyclic_primitive(task.ipoint, task.ispace, m_4d, i, i + 2) for i in (0,1));
    return m_4d[*upper, *lower];
}

def linear_distribute(Task task) {
    return m_1d[task.ipoint[0]];
}

def same_point(Task task) {
    return m_2d[*task.parent.processor(m_2d)]; # same point as parent
}

m_omp_2d = Machine(OMP);

def auto2d_omp(Task task) {
    m_3d = m_omp_2d.auto_split(0, task.ispace);
    upper = tuple(block_primitive(task.ipoint, task.ispace, m_3d, i, i) for i in (0,1));
    return m_3d[*upper, 0];
}

# task_1, task_2, task_3: (gridX * 2, gridY * 2)
IndexTaskMap task_1,task_2,task_3,taco_fill,taco_validate auto2d;
# task_5: (2, 2) SAME_ADDRESS, default huersitic is fine
# init_cublas: 1-dim, over all GPUs
IndexTaskMap init_cublas linear_distribute;
# task7: (gridX, gridY)
IndexTaskMap task_7 auto2d_omp;
# task_4 originate from task_5 (GPUs)
SingleTaskMap task_4 same_point;
# task_6 originate from task_7 (1 OMP)