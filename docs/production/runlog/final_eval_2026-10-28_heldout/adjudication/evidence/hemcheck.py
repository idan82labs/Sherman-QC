"""Distance of each adjudicated hem label line from the measured crown apex line (hem2_out.json)."""
import json, sys, numpy as np
R=sys.argv[1]; H=json.load(open(R+'/adjudication/evidence/hem2_out.json'))
hems={'gb_1baa22':['1baa22_hem'],'gb_d11e33':['d11e33_hem'],'gb_231669':['231669_hemA','231669_hemB'],
      'gb_ac2239':['ac2239_hemYm','ac2239_hemYp'],'gb_68cd06':['68cd06_hem10','68cd06_hem9'],'gb_959f4a':['959f4a_hemL','959f4a_hemR']}
out={}
for g,hs in hems.items():
    L=json.load(open(f'{R}/adjudication/{g}/lines/lines.json'))
    hb=[b for b in L['bends'] if 'hem (' in b.get('note','')]
    for h,b in zip(hs,hb):
        c0,c1=map(np.array,H[h]['crown_line']); a=(c1-c0)/np.linalg.norm(c1-c0)
        if b.get('status')!='ok': out[h]={'bend_id':b['bend_id'],'status':b.get('status'),'error':b.get('error')}; print(g,h,b['bend_id'],'FAILED',b.get('error')); continue
        s=np.array(b['label_line']['start']); e=np.array(b['label_line']['end']); d=(e-s)/np.linalg.norm(e-s)
        def perp(p): v=p-c0; return float(np.linalg.norm(v-a*(v@a)))
        ang=float(np.degrees(np.arccos(min(1,abs(d@a)))))
        o=np.array(H[h]['o']); n=np.array(H[h]['n'])
        mids=(s+e)/2-c0; off_o=float(mids@o); off_n=float(mids@n)
        out[h]={'bend_id':b['bend_id'],'perp_start_mm':round(perp(s),3),'perp_end_mm':round(perp(e),3),'axis_angle_deg':round(ang,3),
                'mid_offset_out_mm':round(off_o,3),'mid_offset_normal_mm':round(off_n,3),'length_mm':round(b['length_mm'],1),
                'crown_len_mm':round(H[h]['crown_len'],1),'tool_bend_angle_deg':round(b['bend_angle_deg'],1),'confidence':b.get('confidence')}
        print(g,h,out[h])
json.dump(out,open(R+'/adjudication/evidence/hemcheck.json','w'),indent=1)
