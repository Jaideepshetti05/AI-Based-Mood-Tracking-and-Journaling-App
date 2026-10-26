import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../api_client.dart';

const moods = <String>['happy', 'sad', 'stressed', 'anxious', 'angry', 'calm', 'excited', 'neutral'];

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});
  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  String? selectedMood;
  final journalController = TextEditingController();
  final api = ApiClient();
  bool saving = false;
  bool analysisEnabled = true;

  @override
  void initState() {
    super.initState();
    _loadSettings();
  }

  Future<void> _loadSettings() async {
    final preferences = await SharedPreferences.getInstance();
    if (mounted) setState(() => analysisEnabled = preferences.getBool('analysis_enabled') ?? true);
  }

  @override
  void dispose() {
    journalController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => Scaffold(
        appBar: AppBar(title: const Text('Mood Journal'), actions: [
          IconButton(onPressed: () => _showPrivacy(context), icon: const Icon(Icons.settings_outlined), tooltip: 'Privacy'),
        ]),
        body: ListView(padding: const EdgeInsets.all(20), children: [
          Text('Good day', style: Theme.of(context).textTheme.headlineMedium),
          Text(DateFormat.yMMMMEEEEd().format(DateTime.now())),
          const SizedBox(height: 24),
          Text('How are you feeling?', style: Theme.of(context).textTheme.titleLarge),
          const SizedBox(height: 12),
          Wrap(spacing: 8, runSpacing: 8, children: moods.map((mood) => ChoiceChip(
                label: Text(mood[0].toUpperCase() + mood.substring(1)),
                selected: selectedMood == mood,
                onSelected: (_) => setState(() => selectedMood = mood),
              )).toList()),
          const SizedBox(height: 24),
          TextField(
            controller: journalController, maxLines: 5, maxLength: 10000,
            onChanged: (_) => setState(() {}),
            decoration: const InputDecoration(labelText: 'Write about your day', border: OutlineInputBorder()),
          ),
          const SizedBox(height: 8),
          FilledButton.icon(
            onPressed: saving || selectedMood == null || journalController.text.trim().isEmpty ? null : _saveEntry,
            icon: saving ? const SizedBox(width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2)) : const Icon(Icons.auto_awesome),
            label: Text(saving ? 'Saving...' : 'Analyze and save'),
          ),
          const SizedBox(height: 24),
          Card(child: ListTile(
            leading: const Icon(Icons.insights_outlined),
            title: const Text('Your insights'),
            subtitle: const Text('Weekly and monthly patterns appear after you save entries.'),
            onTap: () => _showInsights(context),
          )),
        ]),
      );

  Future<void> _saveEntry() async {
    setState(() => saving = true);
    try {
      final entry = await api.createEntry(text: journalController.text.trim(), mood: selectedMood!, analyze: analysisEnabled);
      if (!mounted) return;
      journalController.clear();
      setState(() => selectedMood = null);
      final analysis = entry.analysis;
      if (analysis != null) {
        await showDialog<void>(context: context, builder: (_) => AlertDialog(
          title: const Text('Your entry was analyzed'),
          content: Text('You selected "${entry.mood}". The AI estimate is "${analysis.emotion}" with ${analysis.sentiment} sentiment.\n\n${analysis.limitation}'),
          actions: [TextButton(onPressed: () => Navigator.pop(context), child: const Text('Done'))],
        ));
      }
    } catch (error) {
      if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('$error')));
    } finally {
      if (mounted) setState(() => saving = false);
    }
  }

  void _showPrivacy(BuildContext context) => showDialog<void>(
        context: context,
        builder: (_) => const AlertDialog(
          title: Text('Privacy and settings'),
          content: Text('Your entries are sent to the configured local backend for analysis. Export and delete controls are available through the API. Predictions are estimates, not diagnoses.'),
        ),
      );

  void _showInsights(BuildContext context) => showDialog<void>(
        context: context,
        builder: (_) => const AlertDialog(
          title: Text('Insights'),
          content: Text('Open /api/insights/weekly or /api/insights/monthly from the connected client to view metrics based on saved records.'),
        ),
      );
}
