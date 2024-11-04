# IPCV-ManOverBoard
## Project: Man Overboard

## Abstarct 
This report presents methods for detecting and tracking a person in a video, specifically for a "man overboard" scenario. In such rescue missions, maintaining a clear track of the individual amidst challenging sea conditions is crucial. This work explores advanced algorithms to accurately detect and follow the person of interest, addressing the complexities of object tracking in a dynamic maritime environment.


In this project, we developed a systematic approach to address the challenges of tracking a buoy in a dynamic, wavy sea environment. Through pre-processing techniques, we stabilized video footage to reduce motion distortions, establishing a more reliable basis for accurate tracking. By defining a Region of Interest (ROI) and employing object detection methods within it, we achieved focused and efficient buoy tracking, further enhanced by identifying the horizon line to assist in distance estimation.

Our performance evaluation demonstrated that the tracking model achieved an overall success rate of 86.98%, affirming the efficacy of the tracking approach. However, occasional missed frames, totaling 74, contributed to a modest decrease in the Intersection over Union (IoU) metric. The ground truth annotations, used to compare estimated buoy positions, provided a basis for validating model accuracy and revealed a good match in distance estimation, confirming the potential of the proposed methods for real-world applications. The findings underscore the project’s importance in enhancing object detection and tracking within challenging environments, such as maritime search and rescue. Future applications could use these techniques in automated buoy or person-in water tracking systems to aid in timely rescues, potentially improving safety outcomes in ”man overboard” scenarios.
