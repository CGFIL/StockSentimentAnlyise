import third.thulac as th

thu1 = th.thulac(model_path='../third/models/')
text = thu1.cut("痴人说梦，什么时候华为和京东方合作了", text=True);
print(text)