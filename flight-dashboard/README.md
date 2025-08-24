# Flight Scheduler Dashboard

A modern, responsive dashboard for the Agentic AI Flight Scheduling System built with Next.js, React, and Tailwind CSS.

## Features

- **Real-time Monitoring**: Live system status and flight tracking
- **Interactive Analytics**: Charts and visualizations for flight data
- **AI-Powered Predictions**: Machine learning-based delay predictions
- **What-If Scenarios**: Simulate different operational scenarios
- **Alert Management**: Real-time alerts and notifications
- **Dark/Light Theme**: Automatic theme switching
- **Responsive Design**: Works on desktop, tablet, and mobile

## Tech Stack

- **Framework**: Next.js 14 with App Router
- **UI Library**: React 18 with TypeScript
- **Styling**: Tailwind CSS with custom components
- **Animations**: Framer Motion
- **Charts**: Recharts
- **Icons**: Lucide React

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn
- Python backend API running on port 8000

### Installation

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm run dev
```

3. Open [http://localhost:3000](http://localhost:3000) in your browser

### Production Build

```bash
npm run build
npm start
```

## API Integration

The dashboard connects to the Python backend API running on `http://localhost:8000`. Make sure the backend is running before starting the dashboard.

### Available Endpoints

- `/system/status` - System health and metrics
- `/analytics/peak-analysis` - Peak traffic analysis
- `/alerts/active` - Active system alerts
- `/predictions/delay-risks` - AI delay predictions

## Configuration

Environment variables can be set in `.env.local`:

- `NEXT_PUBLIC_API_URL` - Backend API URL (default: http://localhost:8000)
- `NEXT_PUBLIC_APP_NAME` - Application name
- `NEXT_PUBLIC_APP_VERSION` - Application version

## Dashboard Sections

### Overview
- System status cards
- Peak traffic analysis chart
- Active alerts summary

### Analytics
- Detailed flight analytics
- Performance metrics
- Historical trends

### Optimization
- Schedule optimization recommendations
- AI-powered suggestions

### Prediction
- Delay risk predictions
- ML confidence scores
- Risk factors analysis

### What-If
- Scenario simulation
- Impact analysis
- Alternative scheduling

### Alerts
- Alert management
- Notification settings
- Alert history

## Customization

The dashboard uses Tailwind CSS for styling. You can customize:

- Colors in `tailwind.config.js`
- Global styles in `app/globals.css`
- Component styles inline with Tailwind classes

## Development

### Project Structure

```
flight-dashboard/
├── app/
│   ├── globals.css
│   ├── layout.tsx
│   └── page.tsx
├── public/
├── package.json
├── tailwind.config.js
├── tsconfig.json
└── next.config.js
```

### Adding New Features

1. Create new components in the main page file
2. Add new API endpoints in the fetch functions
3. Update the tab navigation for new sections
4. Add corresponding styles and animations

## Performance

The dashboard is optimized for performance with:

- Server-side rendering with Next.js
- Lazy loading of components
- Optimized animations with Framer Motion
- Efficient state management
- Responsive images and assets

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is part of the Agentic AI Flight Scheduling System.