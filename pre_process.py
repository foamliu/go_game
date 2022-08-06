import argparse
import os

import numpy as np
from tqdm import tqdm

from config import DATA_DIR, chunksize, feature_file_base, label_file_base
from dlgo.encoders.betago import BetaGoEncoder
from dlgo.encoders.oneplane import OnePlaneEncoder
from dlgo.encoders.sevenplane import SevenPlaneEncoder
from dlgo.encoders.simple import SimpleEncoder
from dlgo.goboard_fast import Board, GameState, Move
from dlgo.gosgf import Sgf_game
from dlgo.gotypes import Player, Point
from utils import ensure_folder


def get_handicap(sgf):
    go_board = Board(19, 19)
    first_move_done = False
    move = None
    game_state = GameState.new_game(19)
    if sgf.get_handicap() is not None and sgf.get_handicap() != 0:
        for setup in sgf.get_root().get_setup_stones():
            for move in setup:
                row, col = move
                go_board.place_stone(Player.black, Point(row + 1, col + 1))
        first_move_done = True
        game_state = GameState(go_board, Player.white, None, move)
    return game_state, first_move_done


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--encoder', type=str, default='oneplane', help='encoder')
    parser.add_argument('--compressed', type=bool, default=True, help='compressed')

    args = parser.parse_args()

    data_dir = 'data'
    ensure_folder(data_dir)

    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.sgf')]
    print('len(files): ' + str(len(files)))

    if args.encoder == 'oneplane':
        encoder = OnePlaneEncoder(board_size=(19, 19))
    elif args.encoder == 'sevenplane':
        encoder = SevenPlaneEncoder(board_size=(19, 19))
    elif args.encoder == 'simple':
        encoder = SimpleEncoder(board_size=(19, 19))
    elif args.encoder == 'betago':
        encoder = BetaGoEncoder(board_size=(19, 19))
    else:
        raise ValueError('encoder {} is not supported'.format(args.encoder))

    features = []
    labels = []
    chunk = 0

    num_files = len(files)
    for i in tqdm(range(num_files)):
        file = files[i]
        filename = os.path.join(DATA_DIR, file)

        with open(filename, 'r', encoding='latin-1', errors='ignore') as fp:
            sgf_contents = fp.read().replace('\n', '')

        board = Board(19, 19)
        board.filename = file
        game_state = GameState(board, Player.black, None, None)

        try:
            sgf = Sgf_game.from_string(sgf_contents)
            if sgf.get_handicap() is not None:
                continue

            for item in sgf.main_sequence_iter():  # <5>
                color, move_tuple = item.get_move()
                point = None
                if color is not None:
                    if move_tuple is not None:  # <6>
                        row, col = move_tuple
                        point = Point(row + 1, col + 1)
                        move = Move.play(point)
                    else:
                        move = Move.pass_turn()  # <7>
                    if point is not None:
                        board_tensor = encoder.encode(game_state)  # <8>
                        point_index = encoder.encode_point(point)  # <9>

                        features.append(board_tensor)
                        labels.append(point_index)

                    game_state = game_state.apply_move(move)  # <10>

        except (ValueError, TypeError):
            pass
            # print(("Invalid SGF data, skipping game record %s" % (filename,)))
            # print(("Board was:\n%s" % (board,)))

        # print(len(features))

        while len(features) >= chunksize:  # <1>
            feature_file = feature_file_base % chunk
            label_file = label_file_base % chunk
            chunk += 1
            current_features, features = features[:chunksize], features[chunksize:]
            current_labels, labels = labels[:chunksize], labels[chunksize:]  # <2>

            if args.compressed:
                np.savez_compressed(feature_file, features=current_features)
                np.savez_compressed(label_file, labels=current_labels)
            else:
                np.save(feature_file, current_features)
                np.save(label_file, current_labels)

            # print(feature_file)
            # print(label_file)
