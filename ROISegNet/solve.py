import os,sys
sys.path.append("../caffe")
sys.path.append("../caffe/python")
sys.path.append("../caffe/python/caffe")

sys.path.insert(0, "../../fcn_python/")
sys.path.insert(0, "../../python_layers/")


import caffe
import surgery

import numpy as np


weights      = sys.argv[1]
solver_proto = sys.argv[2]
gpu_id       = np.int(sys.argv[3])

caffe.set_device(gpu_id)
caffe.set_mode_gpu()

solver = caffe.SGDSolver(solver_proto)
solver.net.copy_from(weights)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

for _ in range(20):
    solver.step(10000)


