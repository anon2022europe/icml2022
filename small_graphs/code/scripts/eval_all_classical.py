from ggg.experiments.eval_classical import ex

for g in ["BA", "Gnp"]:
    for ds in ["qm9", "community", "chordal9"]:
        ex.run(named_configs=[ds], config_updates=dict(hpars=dict(model=g)))
