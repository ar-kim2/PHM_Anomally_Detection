# PHM_Anomally_Detection

The approach to automated real-time detection of battery anomalies in distinction from battery aging based on online estimation of battery SOHs and time-varying battery model parameters.<b> 

* Approach to online SOH prediction under inter- and intra-cycle variations of discharging current as well as non-standard charging and discharging practices. 
* Identified in real-time time-varying parameters of a battery electric circuit model by estimating SOC-OCV curve at a cycle with the predicted SOH. 
* Apply the covariance projection filter to two sources of battery terminal voltages, one from actual measurements and the other from the battery electric model updated with time, such that battery anomalies are detected based on a well-defined statistical hypothesis testing with confidence. Experimental results verify that the proposed approach is effective and viable for real-world applications.<br>

# Battery Physical Model

In order to detect anomaly, we need to know the battery state considered as normal at the time of detection. To figure out the range of normal behavior at the time of detection, we may resort to solving the state equation of a battery derived from its physical model. <br>
This figure shows an equivalent circuit model of a lithium-ion battery. <br>
<img width='50%' height='50%' src='https://user-images.githubusercontent.com/60689555/233287966-5ad0f881-e7b4-41dd-905d-f24836f01734.png'> <br>

# CPF Filter for Anomaly Detection

The Covariance Projection Filter has been proposed as a unified framework of data fusion. <br>
In CPF, the predicted state and the measurement are concatenated into the joint vector $Xˆ$ in their joint space, as illustrated in figure. 
Then, the joint probability distribution associated with a joint vector is defined in the joint space, while the constraints existing among individual variables are combined into a constraint manifold in the joint space. 
<br>
<img src='https://user-images.githubusercontent.com/60689555/233290417-f634403b-2c9a-47db-bdfc-25ab7025406b.png'><br>

In this process, the distance between each data is calculated based on the convergence point, and if this distance value is out of the confidence interval, it is classified as an anomaly.

# Reference

Bakr, Muhammad Abu, and Sukhan Lee. "A general framework for data fusion and outlier removal in distributed sensor networks." 2017 IEEE International Conference on Multisensor Fusion and Integration for Intelligent Systems (MFI). IEEE, 2017.
