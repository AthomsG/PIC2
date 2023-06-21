import os
import click
import traceback
from dqn import dqn
from agent import Agent
from estimator import get_estimator
from util import train_path
from wrapper import atari_env
import datetime

@click.command()
@click.option('--game_name', prompt='game name:')
@click.option('--lr', type=float, default=0.0001)
@click.option('--update_target_every', type=int, default=2500)
@click.option('--model_name', default='rqn')
@click.option('--reg', default='none')
@click.option('--lambd', type=float, default=0)
def main(game_name, lr, update_target_every, model_name, reg, lambd):
    assert 'NoFrameskip-v4' in game_name
    now = datetime.datetime.now()
    time = now.strftime('%Y{d}%m{d}%dT%H{d}%M{d}%S'
                        ''.format(d='-', dtd='T'))
    basename = '{}_{}_{}_{}_{}'.format(
        game_name[:-14], model_name, reg, lambd, time)
    #basename = '{}:lr={}:na={}:ute={}:{}'.format(
    #    game_name[:-14], lr, num_agents, update_target_every, model_name)

    env = Agent(1, game_name, basename)
    try:
        estimator = get_estimator(
            model_name, env.action_n, lr, 0.99, reg=reg, lambd=lambd)
        base_path = os.path.join(train_path, basename)
        print("start training!!")
        dqn(env,
            estimator,
            base_path,
            batch_size=32,
            epsilon=0.01,
            save_model_every=1000,
            update_target_every=update_target_every,
            learning_starts=5000,
            memory_size=100000,
            num_iterations=50000000)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt!!")
    except Exception:
        traceback.print_exc()
    finally:
        pass

if __name__ == "__main__":
    main()
