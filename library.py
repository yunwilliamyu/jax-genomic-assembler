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
