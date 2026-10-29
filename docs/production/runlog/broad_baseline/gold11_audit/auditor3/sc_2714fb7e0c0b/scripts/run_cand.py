import sys
sys.path.insert(0,'/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a3_2714')
exec(open('/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a3_2714/cand.py').read())
sys.argv=['x','SF_L','SF_R']
src=open('/private/tmp/claude-501/-Users-idantal/db065cf7-f766-4885-b217-ed8d7edd7ade/scratchpad/a3_2714/extent.py').read()
src=src.replace("from xs import *","from xs import *\nimport xs as _xs\nGOLD=_xs.GOLD")
exec(src)
