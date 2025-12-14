# Latent-Sound-Atlas
Scholarly Project with HONR 46400 - John Martinson Honors College at Purdue University 



<a id="readme-top"></a>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h1 align="center">Latent Sound Atlas</h1>
</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#features">Features</a></li>
        <li><a href="#concept-overview">Concept Overview</a></li>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#supercollider-integration">SuperCollider Integration</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#interaction">Interaction</a></li>
    <li><a href="#current-status">Current Status</a></li>
    <li><a href="#inspiration-and-context">Inspiration & Context</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

This project is an interactive audiovisual system that maps sound samples into a three-dimensional latent space and allows users to explore, select, and audition sounds spatially. The project combines audio feature analysis, clustering, OpenGL visualization, and real-time sound synthesis via SuperCollider. 

This project was developed as a scholarly/creative exploration of how high-dimensional sound features can be navigated visually and experienced sonically. 

### Features
* 3D OpenGL visualization of sound embeddings
* Spatial exploration of clustered sound points
* Real-time OSC communication with SuperCollider
* Dynamic sound playback based on selection
* JSON-driver data pipeline (features, clusters, metadata)
* Modular C++ architecture (rendering, data, OSC, audio)

### Concept Overview
* Audio Analysis (Python): Audio files are analyzed to extract features
* Dimensionality Reduction & Clustering: PCA and clustering algorithms map sounds into a latent 3D space
* Visualization (C++/OpenGL): Each sound become a point in space, color-coded by cluster
* Sound Interaction (OSC + SuperCollider): Selecting or approaching a point triggers sound playback or synthesis

## Built With
### Dependencies
This project vendors all required libraries in `/lib`. 

No external installs required beyond CMake and a C++17 compiler.

### Core
- C++ 17
- OpenGL
- GLFW / GLAD / GLM
- `nlohmann::json`

### Audio
- SuperCollider
- OSC ([oscpp](https://github.com/kaoskorobase/oscpp))

### Data Processing
- Python
- NumPy / scikit-learn
- PCA & Clustering


<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

Follow these instructions to build and run the project locally. 

### Prerequisites
You need the following installed:
* CMake (3.16+)
* Visual Studio, or equivalent compiler
* OpenGL-capable GPU
* SuperCollider

### SuperCollider Integration
The application send OSC message to SuperCollider on:
```
Host: 127.0.0.1
Port: 57120
```
Example OSC message:
```
/play_sound, "synthName", freq, amp
```
SuperCollider must be running and listening for OSC messages before launching the visualization.

### Installation
1. Clone the repository
  ```sh
  git clone https://github.com/bhavyaven/Latent-Sound-Atlas.git
  ```
2. Navigate to the project directory
  ```sh
  cd Latent-Sound-Atlas
  ```
3. Create `out/build` directory and run exectutable from build directory
  ```sh
  mkdir out/build
  cd out/build
  cmake ../..
  cmake --build .
  ```
4. Navigate 3D rendering using Controls listed in accompanying Terminal window.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- USAGE EXAMPLES -->
## Interaction
* Navigate through the 3D space using camera controls
* Hover or select points to trigger sound playback
* Clusters reveal patterns in sound similarity
* Distance and position influence audio parameters

## Current Status
* Visualization pipeline implemented
* OSC communication functional
* JSON-driven data loading
* Ongoing: tuning of synthesis and interaction mapping

## Inspiration and Context
* Sound generation using skills covered in HONR 464: Music Coding
* Latent space visualization in machine learning
* Audiovisual art installations
* Fantasy video game ambience/aesthetics

### Result:
<!-- ![alt text](<Screenshot 2025-10-22 170758.png>) -->

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- ROADMAP -->
## Roadmap
- Phase 1: Data & Analysis
- Phase 2: Visualization Core
- Phase 3: Audio Interaction
- Phase 4: Expressive Mapping & Polish

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTACT -->
## Contact

Bhavya Venkataraghavan - bhavya.v04@gmail.com   
LinkedIn: http://www.linkedin.com/in/bhavya-venkat    
GitHub: https://github.com/bhavyaven   

Project Link: [https://github.com/bhavyaven/Latent-Sound-Atlas#](https://github.com/bhavyaven/Latent-Sound-Atlas#)


<p align="right">(<a href="#readme-top">back to top</a>)</p>

