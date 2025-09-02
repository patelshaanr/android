# Action for Animals Android App

The Android version of **Action for Animals**, an app that empowers users to participate in animal advocacy campaigns and track their actions to support animal welfare.

## Features
- Browse and participate in animal advocacy campaigns
- Track actions taken to support animal welfare
- Interactive, user-friendly interface
- Backend integration with Firestore for user data management
- Custom API endpoints for app interactions

## Basic Structure
- **MainActivity**: Loads campaigns and displays them to the user.
- **CampaignActivity**: Displays details for a selected campaign and allows users to log actions.
- **UserStats**: Uses a local SQLite database to store user stats and progress.
- **SettingsActivity**: Stores preferences in SharedPreferences.

## Development
- Open the project in Android Studio and run on an emulator or device.
- Backend integration is via Firestore and custom API endpoints.

## Testing
- Use Android Studio’s emulator or a connected device for testing.
- Instrumentation tests can be run with:

```bash
./gradlew connectedAndroidTest

```




