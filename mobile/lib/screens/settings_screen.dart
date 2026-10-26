import 'dart:io';

import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';

import '../api_client.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});
  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  final api = ApiClient();
  bool analysisEnabled = true;
  bool busy = false;

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
  Widget build(BuildContext context) => Scaffold(
        appBar: AppBar(title: const Text('Privacy and settings')),
        body: ListView(padding: const EdgeInsets.all(16), children: [
          SwitchListTile(title: const Text('Optional AI analysis'), subtitle: const Text('Send journal text to the configured backend for analysis.'), value: analysisEnabled, onChanged: (value) async {
            final preferences = await SharedPreferences.getInstance();
            await preferences.setBool('analysis_enabled', value);
            if (mounted) setState(() => analysisEnabled = value);
          }),
          const ListTile(title: Text('Data processing'), subtitle: Text('This local-first version stores entries in SQLite on the backend. It does not claim encryption or multi-user authentication.')),
          OutlinedButton.icon(onPressed: busy ? null : _export, icon: const Icon(Icons.download), label: const Text('Export my data')),
          OutlinedButton.icon(onPressed: busy ? null : _deleteAll, icon: const Icon(Icons.delete_forever), label: const Text('Delete all my data')),
        ]),
      );

  Future<void> _export() async {
    setState(() => busy = true);
    try {
      final csv = await api.exportData();
      final directory = await getTemporaryDirectory();
      final file = File('${directory.path}${Platform.pathSeparator}mood-journal.csv');
      await file.writeAsString(csv, flush: true);
      await Share.shareXFiles([XFile(file.path)], text: 'Mood Journal export');
      if (mounted) ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Export shared.')));
    } catch (error) {
      if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('$error')));
    } finally { if (mounted) setState(() => busy = false); }
  }

  Future<void> _deleteAll() async {
    final confirmed = await showDialog<bool>(context: context, builder: (_) => AlertDialog(
      title: const Text('Delete all data?'), content: const Text('This cannot be undone.'), actions: [
        TextButton(onPressed: () => Navigator.pop(context, false), child: const Text('Cancel')),
        FilledButton(onPressed: () => Navigator.pop(context, true), child: const Text('Delete')),
      ],
    ));
    if (confirmed != true) return;
    setState(() => busy = true);
    try { await api.deleteAllData(); if (mounted) ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('All data deleted.'))); }
    catch (error) { if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('$error'))); }
    finally { if (mounted) setState(() => busy = false); }
  }
}
