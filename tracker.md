# Own Features
>< = smaller, less \
>\> = bigger, more \
> y1-y5 = y1 to y5

M7: Mouth corner to eye corner | Happy | (euc_dist(48,36) + euc_dist(54,45))/2, <value = >Happy\
M8: Lower Lip closer to Lower Nose | Sad | mean(y55-y59) - mean(y31-y35), <value = >Sad\
M9: Brow slope | Angry | mean(y21,y22) - mean(y17, y26), >value = >Angry\
M10: Higher Lip closer to Lower Nose | Disgust | mean(y49-y53) - mean(y31-y35), <value= >disgusted\
M11: Brow shape | Surprised or fearful | mean hight between brow extremeties minus the hight of themid point of the brow, averaging with both brows, mean(y17, y21) − y19
/>value = >round = >surprised | <value = >flat = >disgusted. Points used: 17, 19, 21, 22, 24, 26. \
Suprised has round brows, Fearfull has flat brows

### References:
Happy: https://www.youtube.com/watch?v=P_vJoXXgtmY \
Disgusted: https://www.youtube.com/watch?v=HRmbY6R0KK4 \
Surprised or fearful: https://www.youtube.com/watch?v=5V-AQjLyJ7k 

# AI Use
- Helped create the plan used to complete the project, in order to have a clear view and organized work devition between the team members.

># IMPORTANT: 
>### The app must run standalone with python app.py and import its modules from its own folder. For the same reason, keep normalise.py and features.py free of import config, or they break when copied into app/.
>### For delivery, write requirements.txt with the same package names pinned as name==version, taking the versions from pip freeze. Don't paste the whole freeze: it has about 100 transitive packages and a -e git+https://github.com/... line pointing at your repo.
>### For the 2.1 check if we need to explain the mean of each actors mouth widths