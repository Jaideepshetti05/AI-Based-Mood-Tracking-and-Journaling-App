import 'package:flutter/material.dart';

class OnboardingScreen extends StatelessWidget {
  const OnboardingScreen({super.key, required this.onFinished});
  final VoidCallback onFinished;

  @override
  Widget build(BuildContext context) => Scaffold(
        body: SafeArea(child: Padding(padding: const EdgeInsets.all(28), child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Spacer(),
            Icon(Icons.auto_awesome, size: 64, color: Theme.of(context).colorScheme.primary),
            const SizedBox(height: 24),
            Text('A calmer way to reflect', style: Theme.of(context).textTheme.headlineMedium),
            const SizedBox(height: 16),
            const Text('Record your mood, write privately, and review patterns over time. AI predictions are estimates to help you reflect, not medical diagnoses.'),
            const SizedBox(height: 16),
            const Text('Your journal is sent to the configured backend for analysis. This local-first version does not provide encryption or multi-user authentication.'),
            const Spacer(),
            SizedBox(width: double.infinity, child: FilledButton(onPressed: onFinished, child: const Text('Get started'))),
          ],
        ))),
      );
}
