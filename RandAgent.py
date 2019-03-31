# random agent playing game

from utils import *

def show_random():
  mp4list = glob.glob('random/*.mp4')
  if len(mp4list) > 0:
    mp4 = mp4list[0]
    video = io.open(mp4, 'r+b').read()
    encoded = base64.b64encode(video)
    ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
  else: 
    print("Could not find video")

def wrap_env(env):
  env = Monitor(env, 'random', force=True)
  return env

env = wrap_env(gym.make('BreakoutDeterministic-v4'))
print(env.action_space)

# Reset it, returns the starting frame
frame = env.reset()
# Render
env.render()

is_done = False

while not is_done:
  # Perform a random action, returns the new frame, reward and whether the game is over
  frame, reward, is_done, _ = env.step(env.action_space.sample())
  # Render
  env.render()
    
env.close()
show_random()
