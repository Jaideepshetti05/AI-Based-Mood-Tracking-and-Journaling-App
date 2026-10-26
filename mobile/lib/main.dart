import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'screens/home_screen.dart';
import 'screens/history_screen.dart';
import 'screens/insights_screen.dart';
import 'screens/settings_screen.dart';
import 'screens/onboarding_screen.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final preferences = await SharedPreferences.getInstance();
  runApp(MoodJournalApp(showOnboarding: !(preferences.getBool('onboarding_complete') ?? false), preferences: preferences));
}

class MoodJournalApp extends StatelessWidget {
  const MoodJournalApp({super.key, required this.showOnboarding, required this.preferences});
  final bool showOnboarding;
  final SharedPreferences preferences;

  @override
  Widget build(BuildContext context) => MaterialApp(
        title: 'Mood Journal',
        debugShowCheckedModeBanner: false,
        theme: ThemeData(colorSchemeSeed: const Color(0xff6750a4), useMaterial3: true),
        home: RootScreen(showOnboarding: showOnboarding, preferences: preferences),
      );
}

class RootScreen extends StatefulWidget {
  const RootScreen({super.key, required this.showOnboarding, required this.preferences});
  final bool showOnboarding;
  final SharedPreferences preferences;

  @override
  State<RootScreen> createState() => _RootScreenState();
}

class _RootScreenState extends State<RootScreen> {
  late bool showOnboarding = widget.showOnboarding;

  @override
  Widget build(BuildContext context) => showOnboarding
      ? OnboardingScreen(onFinished: () {
          widget.preferences.setBool('onboarding_complete', true);
          setState(() => showOnboarding = false);
        })
      : const AppShell();
}

class AppShell extends StatefulWidget {
  const AppShell({super.key});
  @override
  State<AppShell> createState() => _AppShellState();
}

class _AppShellState extends State<AppShell> {
  int index = 0;
  final screens = const [HomeScreen(), HistoryScreen(), InsightsScreen(), SettingsScreen()];

  @override
  Widget build(BuildContext context) => Scaffold(
        body: screens[index],
        bottomNavigationBar: NavigationBar(
          selectedIndex: index,
          onDestinationSelected: (value) => setState(() => index = value),
          destinations: const [
            NavigationDestination(icon: Icon(Icons.home_outlined), selectedIcon: Icon(Icons.home), label: 'Home'),
            NavigationDestination(icon: Icon(Icons.menu_book_outlined), selectedIcon: Icon(Icons.menu_book), label: 'History'),
            NavigationDestination(icon: Icon(Icons.insights_outlined), selectedIcon: Icon(Icons.insights), label: 'Insights'),
            NavigationDestination(icon: Icon(Icons.settings_outlined), selectedIcon: Icon(Icons.settings), label: 'Settings'),
          ],
        ),
      );
}
