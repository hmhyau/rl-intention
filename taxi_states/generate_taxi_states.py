import gym
from ansi2html import Ansi2HTMLConverter
import imgkit

env = gym.make('Taxi-v3')
conv = Ansi2HTMLConverter(font_size='xx-large', dark_bg=False, scheme='xterm')

for i in range(env.observation_space.n):
  out = env.render(i)
  html = conv.convert(out)
  options = {
    'format' : 'png',
    'crop-w' : 208,
    'crop-h' : 244,
  }
  imgkit.from_string(html, "{}.png".format(str(i).zfill(3)), options=options)

