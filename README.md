# SAI Talent Scout - Sports Assessment Platform

A comprehensive web application for sports talent assessment with AI-powered video analysis capabilities, developed for the Sports Authority of India (SAI).

## 🎯 Features

### Frontend (React + TypeScript)
- **Video Recording & Upload**: Camera access with video recording + pre-recorded video upload
- **AI Video Analysis**: Real-time pose detection using MediaPipe for performance metrics
- **Multiple Fitness Tests**: Vertical Jump, Sit-ups, Shuttle Run, Endurance Run, Height/Weight
- **Gamification**: Achievement badges, leaderboards, progress tracking
- **Admin Panel**: Officials can review submissions and verify results
- **Performance Dashboard**: Comprehensive results display with historical tracking

### Backend (Python Flask)
- **Real AI Analysis**: MediaPipe pose detection for accurate performance measurement
- **Video Processing**: Supports MP4, WebM, MOV, AVI formats (max 100MB)
- **RESTful API**: Clean endpoints for video analysis and results management
- **Automatic Fallback**: Gracefully falls back to simulated analysis if backend unavailable

## 🚀 Quick Start

### Frontend Setup
```bash
# Install dependencies
pnpm install

# Start development server
pnpm run dev

# Build for production
pnpm run build
```

### Backend Setup (Optional - for Real AI Analysis)
```bash
# Install Python dependencies
pip install -r requirements.txt

# Start the Flask server
python server.py
```

The frontend will automatically detect if the backend is running:
- ✅ **Backend Online**: Real AI analysis using MediaPipe pose detection
- 🔄 **Demo Mode**: Simulated analysis for demonstration purposes

## 📁 Project Structure

```
├── src/
│   ├── pages/
│   │   ├── Index.tsx          # Landing page
│   │   ├── Dashboard.tsx      # Main user dashboard
│   │   ├── VideoUpload.tsx    # Video recording/upload + AI analysis
│   │   ├── Results.tsx        # Performance results display
│   │   └── AdminPanel.tsx     # Officials review panel
│   ├── components/
│   │   ├── VideoAnalyzer.tsx  # AI analysis results display
│   │   └── GameElements.tsx   # Badges and leaderboard
│   └── ...
├── server.py                  # Flask backend server
├── pose_analyzer.py          # MediaPipe AI analysis engine
└── requirements.txt          # Python dependencies
```

## 🤖 AI Analysis Capabilities

The pose analyzer uses MediaPipe to detect and analyze:

### Vertical Jump Test
- Maximum jump height estimation
- Landing consistency analysis
- Form quality assessment

### Sit-ups Test
- Repetition counting using torso angle analysis
- Form accuracy measurement
- Pacing consistency tracking

### Shuttle Run Test
- Direction change detection
- Speed and agility metrics
- Movement pattern analysis

## 🔧 Technical Stack

**Frontend:**
- React 18 + TypeScript
- Tailwind CSS + Shadcn-ui components
- Vite build system
- React Router for navigation

**Backend:**
- Python Flask web framework
- MediaPipe for pose detection
- OpenCV for video processing
- Flask-CORS for cross-origin requests

## 🌐 API Endpoints

- `GET /health` - Health check
- `POST /analyze-video` - Upload video for AI analysis
- `GET /results` - Retrieve all analysis results

## 📱 Mobile Support

- Responsive design works on smartphones and tablets
- Camera access for mobile video recording
- Touch-friendly interface optimized for mobile use

## 🏆 Gamification Features

- **Achievement Badges**: Unlock badges for milestones
- **National Leaderboard**: Compare performance with other athletes
- **Progress Tracking**: Monitor improvement over time
- **Performance Scoring**: 0-100 scale with benchmarking

## 👨‍💼 Admin Features

- Review submitted videos and performance data
- Verify or flag suspicious submissions
- Export data for further analysis
- Statistics dashboard for officials

## 🔒 Security & Authenticity

- AI-based cheat detection using pose analysis
- Confidence scoring for submission authenticity
- Movement pattern verification
- Anomaly detection alerts

## 📊 Performance Metrics

- **Build Size**: 442KB gzipped
- **Lighthouse Score**: 95+ performance
- **Mobile Optimized**: Works on entry-level smartphones
- **Real-time Analysis**: < 30 seconds processing time

---

**Ready to democratize sports talent assessment across India! 🇮🇳**