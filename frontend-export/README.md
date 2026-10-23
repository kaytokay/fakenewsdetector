# Veritas Lens frontend export

This folder contains a standalone React + Tailwind dashboard component and the original Stitch HTML export.

## Files
- `VeritasDashboard.jsx`: reusable responsive dashboard component with working modality tabs, claim input, character count, mobile navigation, and accessible status feedback.
- `VeritasDashboard.html`: original Stitch-generated dashboard HTML.

## Integration
Import `VeritasDashboard` into a React app and render it as a page. Tailwind CSS must be configured in the host app. The component uses utility classes only and does not depend on an icon package. It uses React hooks, semantic landmarks, visible focus styles, screen-reader status announcements, keyboard-friendly controls, and responsive layouts.

The example content is illustrative. Connect the form to your verification API and replace the sample checks and metrics with real data before deployment.
