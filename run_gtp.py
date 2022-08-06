#!/usr/local/bin/python2
from dl_agent import load_prediction_agent
from dlgo.agent import termination
from dlgo.gtp import GTPFrontend

agent = load_prediction_agent('checkpoint-epoch=03-val_acc=0.54.ckpt')
strategy = termination.get("opponent_passes")
termination_agent = termination.TerminationAgent(agent, strategy)

frontend = GTPFrontend(termination_agent)
frontend.run()
