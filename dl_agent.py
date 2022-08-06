import numpy as np
import torch
from torch.nn import functional as F
from dlgo import goboard
from dlgo.agent.base import Agent
from dlgo.agent.helpers import is_point_an_eye
from dlgo.encoders.oneplane import OnePlaneEncoder
from models import AlphaZeroModel


class DeepLearningAgent(Agent):
    def __init__(self, model, encoder):
        Agent.__init__(self)
        self.model = model
        self.encoder = encoder

    def predict(self, game_state):
        encoded_state = self.encoder.encode(game_state)
        input_tensor = torch.Tensor([encoded_state])
        output_tensor = self.model(input_tensor)
        output_tensor = F.softmax(output_tensor, dim=-1)
        output_tensor = output_tensor.detach().numpy()
        return output_tensor[0]

    def select_move(self, game_state):
        num_moves = self.encoder.board_width * self.encoder.board_height
        move_probs = self.predict(game_state)

        move_probs = move_probs ** 3  # <1>
        eps = 1e-6
        move_probs = np.clip(move_probs, eps, 1 - eps)  # <2>
        move_probs = move_probs / np.sum(move_probs)  # <3>

        candidates = np.arange(num_moves)  # <1>
        ranked_moves = np.random.choice(
            candidates, num_moves, replace=False, p=move_probs)  # <2>
        for point_idx in ranked_moves:
            point = self.encoder.decode_point_index(point_idx)
            if game_state.is_valid_move(goboard.Move.play(point)) and \
                    not is_point_an_eye(game_state.board, point, game_state.next_player):  # <3>
                return goboard.Move.play(point)
        return goboard.Move.pass_turn()  # <4>

    def serialize(self, h5file):
        pass


def load_prediction_agent(ckpt_path):
    # model = AlphaZeroModel(input_size=(1, 19, 19))
    model = AlphaZeroModel.load_from_checkpoint(checkpoint_path=ckpt_path)

    encoder = OnePlaneEncoder(board_size=(19, 19))
    return DeepLearningAgent(model, encoder)


if __name__ == '__main__':
    agent = load_prediction_agent('checkpoint-epoch=02-val_acc=0.54.ckpt')
    print(agent)
