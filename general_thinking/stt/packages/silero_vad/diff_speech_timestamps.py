from dataclasses import dataclass


@dataclass
class Pt:
  idx: int
  tag: str
  ts: int

  @property
  def is_start(self):
    return self.tag == 'start'

  @property
  def is_end(self):
    return self.tag == 'end'

  @property
  def is_left(self):
    return self.idx == 0
  
  @property
  def is_right(self):
    return self.idx == 1
  

def diff_speech_timestamps(ts0, ts1) -> list[Pt]:
  """Returns two sets of psts, only in 1 and only in 2"""
  def to_pts_array(index, pst):
    arr = []
    for r in pst:
      arr.append(Pt(index, 'start', r['start']))
      arr.append(Pt(index, 'end', r['end']))
    return arr

  arr0 = to_pts_array(0, ts0)
  arr1 = to_pts_array(1, ts1)
  merged = _merge(arr0, arr1)
  diff = []
  leftRunning = False
  rightRunning = False
  aligned = True
  diffStart = None
  for pt in merged:
    if pt.is_start:
      if pt.is_left:
        leftRunning = True
      else:
        rightRunning = True
    else:
      if pt.is_left:
        leftRunning = False
      else:
        rightRunning = False

    aligned = leftRunning == rightRunning 
    if not aligned:
      diffStart = (0 if leftRunning else 1, pt.ts)
    else:
      if diffStart is not None and pt.ts != diffStart[1]:
        diff.append((diffStart[0], (diffStart[1], pt.ts)))
        diffStart = None

  return diff


def _merge(a0, a1):
  """Merge two sorted arrays a0 and a1 by first element of tuple values."""
  merged = []
  idxLeft = 0
  idxRight = 0
  while idxLeft < len(a0) or idxRight < len(a1):
    if idxLeft == len(a0):
      merged.append(a1[idxRight])
      idxRight += 1
      continue
    if idxRight == len(a1):
      merged.append(a0[idxLeft])
      idxLeft += 1
      continue

    if a0[idxLeft].ts < a1[idxRight].ts:
      merged.append(a0[idxLeft])
      idxLeft += 1
    else:
      merged.append(a1[idxRight])
      idxRight += 1

  return merged
