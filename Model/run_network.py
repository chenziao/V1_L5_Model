import sys
import os
import warnings
import synapses
from bmtk.simulator import bionet
from bmtk.simulator.bionet.pyfunction_cache import add_weight_function
from neuron import h

CONFIG = 'config.json'
USE_CORENEURON = False


def run(config_file=CONFIG, use_coreneuron=USE_CORENEURON):

    warnings.simplefilter(action='ignore', category=FutureWarning)

    # register synaptic weight function
    synapses.load(randseed=1111)
    add_weight_function(synapses.lognormal_weight, name='lognormal_weight')

    if use_coreneuron:
        import corebmtk
        conf = corebmtk.Config.from_json(config_file, validate=True)
    else:
        conf = bionet.Config.from_json(config_file, validate=True)

    conf.build_env()
    graph = bionet.BioNetwork.from_config(conf)

    if use_coreneuron:
        sim = corebmtk.CoreBioSimulator.from_config(
            conf, network=graph, gpu=False)
    else:
        sim = bionet.BioSimulator.from_config(conf, network=graph)

    '''
    # This calls insert_mechs() on each cell to use its gid as a seed
    # to the random number generator, so that each cell gets a different
    # random seed for the point-conductance noise
    cells = graph.get_local_cells()
    for cell in cells:
        cells[cell].hobj.insert_mechs(cells[cell].gid)
    '''

    # clear ecp temporary directory to avoid errors
    pc = h.ParallelContext()
    if pc.id() == 0:
        try:
            ecp_tmp = conf['reports']['ecp']['tmp_dir']
        except:
            pass
        else:
            if os.path.isdir(ecp_tmp):
                for f in os.listdir(ecp_tmp):
                    if f.endswith(".h5"):
                        try:
                            os.remove(os.path.join(ecp_tmp, f))
                        except Exception as e:
                            print(f'Failed to delete {f}. {e}')
    pc.barrier()

    sim.run()

    bionet.nrn.quit_execution()


if __name__ == '__main__':
    for i, s in enumerate(sys.argv):
        if s in __file__:
            break

    if i < len(sys.argv) - 1:
        argv = sys.argv[i + 1:]
        for i in range(1, len(argv)):
            argv[i] = eval(argv[i])
        run(*argv)
    else:
        run()
