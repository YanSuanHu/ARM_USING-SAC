import numpy as np
demo_data = np.load("master_data.npz")
obs_demo = demo_data["obs"]
act_demo = demo_data["act"]
next_obs_demo = demo_data["obs_next"]
rewards_demo = demo_data["rewards"]
done_demo = demo_data["done"]
maxi = 0
for i in act_demo:
    for j in i:
        for k in j[3:6]:
            maxi = max(maxi,k)
print(maxi)


