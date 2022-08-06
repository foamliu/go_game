from dl_agent import load_prediction_agent
from dlgo import httpfrontend


def main():
    bind_address = '0.0.0.0'
    port = 5555

    agent = load_prediction_agent('checkpoint-epoch=03-val_acc=0.54.ckpt')
    bots = {'predict': agent}

    web_app = httpfrontend.get_web_app(bots)
    web_app.run(host=bind_address, port=port, threaded=False)


if __name__ == '__main__':
    main()
