from folds import *
fine={}
probe={'F1_upflange_v-54':[(-490,-478),(-275,-262)], 'F2_upflange_v-278':[(-490,-478),(-275,-262)],
       'F5_slotflange_v-85':[(-410,-396),(-326,-310)], 'F6_slotflange_v-258':[(-410,-396),(-326,-310)],
       'F3_leftflange_u-477':[(-276,-262),(-72,-58)], 'F4_endwall_u-224':[(-282,-270),(-62,-50)],
       'F7_ear_v-54':[(-118,-106),(-74,-60)], 'F8_ear_v-278':[(-118,-106),(-74,-60)]}
for name,wins in probe.items():
    fd=FOLDS[name]; out=[]
    for lo,hi in wins:
        for x0 in np.arange(lo,hi+0.01,0.5):
            st=station(fd,x0,half=0.25)
            ok=st is not None and st['ok']
            if ok:
                med={'F1_upflange_v-54':(-54.08,-66.0),'F2_upflange_v-278':(-278.01,-65.85),'F5_slotflange_v-85':(-84.0,-66.35),'F6_slotflange_v-258':(-258.3,-66.24),
                     'F3_leftflange_u-477':(-476.67,-65.85),'F4_endwall_u-224':(-224.22,-66.01),'F7_ear_v-54':(-224.25,-53.81),'F8_ear_v-278':(-223.81,-277.7)}[name]
                dev=float(np.max(np.abs(st['apex']-np.array(med))))
                ok = dev<1.5 and 80<st['raw_angle']<100
            out.append((float(x0),bool(ok)))
    fine[name]=out
    s=''.join('#' if o else '.' for _,o in out)
    print(name, wins, s)
