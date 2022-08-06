from dl_agent import load_prediction_agent
from dlgo.agent.termination import PassWhenOpponentPasses
from dlgo.gtp.play_local import LocalGtpBot

agent = load_prediction_agent('checkpoint-epoch=03-val_acc=0.54.ckpt')

gtp_bot = LocalGtpBot(go_bot=agent, termination=PassWhenOpponentPasses(),
                      handicap=0, opponent='pachi')
gtp_bot.run()
