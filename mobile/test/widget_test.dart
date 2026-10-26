import 'package:flutter_test/flutter_test.dart';
import 'package:mood_journal/main.dart';
import 'package:shared_preferences/shared_preferences.dart';

void main() {
  testWidgets('home screen renders mood controls', (tester) async {
    SharedPreferences.setMockInitialValues({'onboarding_complete': true});
    final preferences = await SharedPreferences.getInstance();
    await tester.pumpWidget(MoodJournalApp(showOnboarding: false, preferences: preferences));
    expect(find.text('How are you feeling?'), findsOneWidget);
    expect(find.text('Happy'), findsOneWidget);
    expect(find.text('Analyze and save'), findsOneWidget);
  });
}
