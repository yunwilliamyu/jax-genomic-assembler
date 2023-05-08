import numpy as np

def int_to_16mer(x):
  '''Converts an integer to a 16-mer. Assumes that the integer is in range [0, 2**32)'''
  def pair_to_base(b):
    '''Converts binary pair to base'''
    if b=='00':
      return 'A'
    elif b=='01':
      return 'C'
    elif b=='10':
      return 'G'
    elif b=='11':
      return 'T'
    else:
      raise ValueError('Invalid binary pair' + str(b))
  x = x % 4**16
  x_bin = str(bin(x))[2:]
  x_bin = '0'*(32-len(x_bin)) + x_bin
  x_bin = list(x_bin)
  return ''.join([pair_to_base(''.join(y)) for y in zip(x_bin[::2], x_bin[1::2])])

def mutate(x, d, types=['I', 'D', 'S'], prng=np.random.RandomState(None)):
  '''Puts in 'd' mutations into a k-mer x (as string of A,C,G,T)
  Will pad random characters to beginning to make things harder if we don't
  end up with a k-mer, forcing the output to be the same length
  
  Note that the actual edit distance for the returned string may not be d

  This is for a couple reasons, including the padding operation we perform, but
  also just because edit distance is weird and not just a sum of (suboptimal) edits
  '''
  x = list(x)
  edit_positions = prng.randint(len(x), size=d)
  edit_types = prng.choice(types, size=d)
  edit_val = prng.choice(['A','C','G','T'], size=d)
  for p, t, v in zip(edit_positions, edit_types, edit_val):
    if t=='I':
      x[p]+=v
    elif t=='D':
      x[p]=x[p][1:]
    elif t=='S':
      x[p]=v

  ans = ''.join(x)

  prepend = ''.join(prng.choice(['A','C','G','T'], size=max(len(x)-len(ans),0) ))
  ans = prepend + ans
  ans = ans[:len(x)]
  return ans

def sample_pairs_16mers(seed=42, sample_estimate=10000):
  '''Actual number will be be approximately sample_estimate for large numbers'''
  prng = np.random.RandomState(seed)
  sample_start = sample_estimate // 2

  XYD = []
  # Initially start by generating random pairs.
  # Notice that random strings are probably only about distance somewhat lower away from each other
  XY = prng.randint(low=0, high=(4**16), size=(sample_start*57//100,2))
  XY = [(int_to_16mer(x), int_to_16mer(y)) for x,y in XY]
  XYD2 = [(x, y, distance(x,y)) for x,y in XY]
  XYD.extend(XYD2)

  # Thus, we need to have a way to sample our 16-mer pairs to be closer to each other
  for d in range(0,16):
    if d>10:
      scale=5
    elif d>5:
      scale=7
    elif d==2:
      scale=2
    elif d==3:
      scale=8
    elif d==1:
      scale=7
    elif d==0:
      scale=11
    else:
      scale=12
    X = prng.randint(low=0, high=(4**16), size=sample_start*scale//100)
    X = [int_to_16mer(x) for x in X]
    Y = [mutate(x, d, prng=prng) for x in X]
    XYD2 = [(x, y, distance(x,y)) for x,y in zip(X,Y)]
    XYD.extend(XYD2)

  #c = Counter([d for _,_,d in XYD])

  # However, because of parity issues with indels, we are going to manually add
  # some more substitution-only examples for distance 1, 3, 5
  for d in [1,3,5]:
    if d==1:
      scale=12
    elif d==3:
      scale=16
    elif d==5:
      scale=3
    X = prng.randint(low=0, high=(4**16), size=sample_start*scale//100)
    X = [int_to_16mer(x) for x in X]
    Y = [mutate(x, d, types=['S'], prng=prng) for x in X]   
    XYD2 = [(x, y, distance(x,y)) for x,y in zip(X,Y)]
    XYD.extend(XYD2) 
  return XYD
