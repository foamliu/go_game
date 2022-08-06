from dlgo.agent.naive import RandomBot
from dlgo.httpfrontend.server import get_web_app

port = 5555

random_agent = RandomBot()
web_app = get_web_app({'random': random_agent})
web_app.run(port=port)
