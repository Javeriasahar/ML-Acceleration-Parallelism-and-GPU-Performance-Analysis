

ML Acceleration: Parallelism and GPU Performance Analysis

Contributors
	•	Neha Rahim (2022481)
	•	Javeria Sahar (2022241)

Overview

With AI models growing chunkier and datasets ballooning, training times are getting dragged. This project dives deep into the real tea: how to speed up training without wrecking accuracy. We explore parallel processing, distributed computing, GPU acceleration, and even hybrid methods to find the fastest, most efficient way to train a Random Forest Classifier. Spoiler: PyTorch on GPU wins.

Tech Stack & Environment
	•	Languages: Python
	•	Libraries: scikit-learn, PyTorch, TensorFlow, Dask, mpi4py, pyspark, numba
	•	Platform: Google Colab (T4 GPU)

Methods Compared

Category	Method	Execution Time
Serial Execution	CPU / GPU	Baseline
Parallel (CPU)	Multithreading	3.61s
	Multiprocessing	7.49s
Distributed (CPU)	Dask	4.64s
	MPI	6.85s
	Spark	9.41s
GPU Acceleration	PyTorch	0.92s
	TensorFlow	0.95s
	CUDA (numba)	2.21s
Hybrid Combos	Dask + Multithreading	1.45s
	Multithreading + PyTorch	4.25s
	PyTorch + Dask	12.54s
	All Three Combined	2.64s

Key Findings
	•	PyTorch on GPU is the undisputed queen—fastest with zero loss in accuracy.
	•	Multithreading outshines multiprocessing on CPUs.
	•	Dask slays in distributed computing for moderate workloads.
	•	Hybrid models? Meh. Often too much overhead, not enough gain.

Challenges Faced
	•	Google Colab’s resource limits were a buzzkill for bigger datasets.
	•	Spark and MPI were too extra for small tasks—high overhead, low payoff.
	•	Hybrid setups turned into debugging nightmares.

Conclusion

If you’re chasing performance in ML training:
	•	Use GPU acceleration with PyTorch if available.
	•	Don’t overcomplicate with hybrids unless you’re dealing with massive data.
	•	Parallelism shines, but only when applied wisely.

Future Directions
	•	Test on larger datasets and deeper models (bring on the neural networks).
	•	Explore asynchronous and pipeline parallelism.
	•	Try multi-GPU or TPU systems.
	•	Add energy consumption analysis for sustainable ML.

