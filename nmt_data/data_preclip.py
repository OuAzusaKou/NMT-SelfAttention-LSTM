source_add="train.en"
target_add="train.vi"
out_sourceadd="train.en.cliped"
out_targetadd="train.vi.cliped"
r_source=open(source_add,'r')
r_target=open(target_add,'r')
w_source=open(out_sourceadd,'w')
w_target=open(out_targetadd,'w')
while True:
    source_line=r_source.readline()
    target_line=r_target.readline()
    if not source_line:
        break
    if (len(source_line)<300) and (len(target_line)<300):
        w_source.write(source_line)
        w_target.write(target_line)
r_source.close()
r_target.close()
w_source.close()
w_target.close()
