`src0`: original
`src1`: disable all index launches
`src2`: original + mapping DSL
    - mapping0: original
    - mapping1: reverse the node dimension for distribute_charge,update_voltages
    - mapping2: reverse the order of processors for distribute_charge,update_voltages
    - mapping3: reverse the order of node only for distribute_charge
`src3`: disable all index launch + mapping DSL