import 'package:flutter/material.dart';
import 'package:intl/intl.dart';

import '../api_client.dart';
import '../models.dart';

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});
  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  final api = ApiClient();
  final search = TextEditingController();
  String? mood;
  late Future<List<JournalEntry>> entries;

  @override
  void initState() {
    super.initState();
    entries = api.getEntries();
  }

  @override
  void dispose() {
    search.dispose();
    super.dispose();
  }

  void refresh() => setState(() => entries = api.getEntries(mood: mood, search: search.text));

  @override
  Widget build(BuildContext context) => Scaffold(
        appBar: AppBar(title: const Text('Journal history')),
        body: Column(children: [
          Padding(
            padding: const EdgeInsets.all(16),
            child: TextField(
              controller: search,
              onChanged: (_) => refresh(),
              decoration: const InputDecoration(prefixIcon: Icon(Icons.search), labelText: 'Search entries', border: OutlineInputBorder()),
            ),
          ),
          SingleChildScrollView(
            scrollDirection: Axis.horizontal,
            padding: const EdgeInsets.symmetric(horizontal: 16),
            child: Row(children: [
              FilterChip(label: const Text('All moods'), selected: mood == null, onSelected: (_) { mood = null; refresh(); }),
              const SizedBox(width: 8),
              ...['happy', 'sad', 'stressed', 'anxious', 'angry', 'calm', 'excited', 'neutral'].map((value) => Padding(
                    padding: const EdgeInsets.only(right: 8),
                    child: FilterChip(label: Text(value), selected: mood == value, onSelected: (_) { mood = value; refresh(); }),
                  )),
            ]),
          ),
          Expanded(child: FutureBuilder<List<JournalEntry>>(
            future: entries,
            builder: (context, snapshot) {
              if (snapshot.connectionState == ConnectionState.waiting) return const Center(child: CircularProgressIndicator());
              if (snapshot.hasError) return _Message(text: 'Could not load history. Check the backend connection.');
              if (snapshot.data!.isEmpty) return const _Message(text: 'No entries yet. Your saved journals will appear here.');
              return RefreshIndicator(onRefresh: () async => refresh(), child: ListView.builder(
                padding: const EdgeInsets.all(16),
                itemCount: snapshot.data!.length,
                itemBuilder: (_, index) => _EntryCard(entry: snapshot.data![index], onEdit: () => _edit(snapshot.data![index]), onDelete: () async {
                  await api.deleteEntry(snapshot.data![index].id);
                  refresh();
                }, onCorrect: () => _correct(snapshot.data![index])),
              ));
            },
          )),
        ]),
      );

  Future<void> _correct(JournalEntry entry) async {
    if (entry.analysis == null) return;
    final emotion = await showDialog<String>(context: context, builder: (_) => SimpleDialog(
      title: const Text('Choose the emotion that fits best'),
      children: ['joy', 'sadness', 'anger', 'fear', 'nervousness', 'neutral'].map((value) => SimpleDialogOption(
        onPressed: () => Navigator.pop(context, value), child: Text(value),
      )).toList(),
    ));
    if (emotion != null) {
      await api.correctEntry(entry.id, helpful: false, emotion: emotion);
      refresh();
    }
  }

  Future<void> _edit(JournalEntry entry) async {
    final controller = TextEditingController(text: entry.text);
    final save = await showDialog<bool>(context: context, builder: (_) => AlertDialog(
      title: const Text('Edit journal entry'),
      content: TextField(controller: controller, maxLines: 5, maxLength: 10000, decoration: const InputDecoration(border: OutlineInputBorder())),
      actions: [
        TextButton(onPressed: () => Navigator.pop(context, false), child: const Text('Cancel')),
        FilledButton(onPressed: () => Navigator.pop(context, true), child: const Text('Save')),
      ],
    ));
    if (save == true && controller.text.trim().isNotEmpty) {
      await api.updateEntry(entry.id, text: controller.text.trim(), mood: entry.mood);
      refresh();
    }
    controller.dispose();
  }
}

class _EntryCard extends StatelessWidget {
  const _EntryCard({required this.entry, required this.onEdit, required this.onDelete, required this.onCorrect});
  final JournalEntry entry;
  final VoidCallback onEdit;
  final VoidCallback onDelete;
  final VoidCallback onCorrect;

  @override
  Widget build(BuildContext context) => Card(
        child: Padding(padding: const EdgeInsets.all(16), child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Row(children: [
            Chip(label: Text(entry.mood)),
            const Spacer(),
            Text(DateFormat.yMMMd().format(entry.createdAt.toLocal())),
            PopupMenuButton<String>(onSelected: (value) { if (value == 'edit') onEdit(); if (value == 'delete') onDelete(); if (value == 'correct') onCorrect(); }, itemBuilder: (_) => [
              const PopupMenuItem(value: 'edit', child: Text('Edit')),
              if (entry.analysis != null) const PopupMenuItem(value: 'correct', child: Text('Correct prediction')),
              const PopupMenuItem(value: 'delete', child: Text('Delete')),
            ]),
          ]),
          Text(entry.text, maxLines: 3, overflow: TextOverflow.ellipsis),
          if (entry.analysis != null) Padding(padding: const EdgeInsets.only(top: 8), child: Text(
            'AI: ${entry.analysis!.emotion} · ${entry.analysis!.sentiment}',
            style: Theme.of(context).textTheme.bodySmall,
          )),
        ]));
}

class _Message extends StatelessWidget {
  const _Message({required this.text});
  final String text;
  @override
  Widget build(BuildContext context) => Center(child: Padding(padding: const EdgeInsets.all(32), child: Text(text, textAlign: TextAlign.center)));
}
