import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';

import '../api_client.dart';
import '../models.dart';

class InsightsScreen extends StatefulWidget {
  const InsightsScreen({super.key});
  @override
  State<InsightsScreen> createState() => _InsightsScreenState();
}

class _InsightsScreenState extends State<InsightsScreen> {
  bool monthly = false;
  final api = ApiClient();

  @override
  Widget build(BuildContext context) => Scaffold(
        appBar: AppBar(title: const Text('Insights')),
        body: FutureBuilder<Insights>(
          future: api.getInsights(monthly: monthly),
          builder: (context, snapshot) {
            if (snapshot.connectionState == ConnectionState.waiting) return const Center(child: CircularProgressIndicator());
            if (snapshot.hasError) return const Center(child: Text('Insights are unavailable while offline.'));
            final data = snapshot.data!;
            return ListView(padding: const EdgeInsets.all(16), children: [
              SegmentedButton<bool>(segments: const [ButtonSegment(value: false, label: Text('7 days')), ButtonSegment(value: true, label: Text('30 days'))],
                selected: {monthly}, onSelectionChanged: (value) => setState(() => monthly = value.first)),
              const SizedBox(height: 16),
              _MetricCard(label: 'Journal entries', value: '${data.entryCount}'),
              _MetricCard(label: 'Most frequent mood', value: data.mostFrequentMood ?? 'Not enough data'),
              _MetricCard(label: 'AI/user agreement', value: data.agreementRate == null ? 'Not measured' : '${(data.agreementRate! * 100).round()}%'),
              _DistributionCard(title: 'Mood distribution', values: data.moodDistribution),
              _DistributionCard(title: 'Sentiment distribution', values: data.sentimentDistribution),
              _DistributionCard(title: 'Emotion predictions', values: data.emotionDistribution),
              const Text('Insights describe stored records and are not medical assessments.'),
            ]);
          },
        ),
      );
}

class _MetricCard extends StatelessWidget {
  const _MetricCard({required this.label, required this.value});
  final String label;
  final String value;
  @override
  Widget build(BuildContext context) => Card(child: ListTile(title: Text(label), trailing: Text(value, style: Theme.of(context).textTheme.titleMedium)));
}

class _DistributionCard extends StatelessWidget {
  const _DistributionCard({required this.title, required this.values});
  final String title;
  final List<Distribution> values;
  @override
  Widget build(BuildContext context) {
    final maxValue = values.fold<int>(1, (current, item) => item.count > current ? item.count : current);
    return Card(child: Padding(padding: const EdgeInsets.all(16), child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Text(title, style: Theme.of(context).textTheme.titleMedium),
      if (values.isNotEmpty) SizedBox(
        height: 180,
        child: BarChart(BarChartData(
          maxY: maxValue.toDouble() + 1,
          barGroups: values.asMap().entries.map((entry) => BarChartGroupData(x: entry.key, barRods: [
            BarChartRodData(toY: entry.value.count.toDouble(), width: 18, borderRadius: BorderRadius.circular(4)),
          ])).toList(),
          titlesData: FlTitlesData(
            leftTitles: const AxisTitles(sideTitles: SideTitles(showTitles: true, reservedSize: 28)),
            rightTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
            topTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
            bottomTitles: AxisTitles(sideTitles: SideTitles(showTitles: true, getTitlesWidget: (value, meta) {
              final index = value.toInt();
              return index >= 0 && index < values.length
                  ? SideTitleWidget(meta: meta, child: Text(values[index].label.substring(0, values[index].label.length > 4 ? 4 : values[index].label.length)))
                  : const SizedBox.shrink();
            })),
          ),
          borderData: FlBorderData(show: false),
          gridData: const FlGridData(show: false),
        )),
      ),
      ...values.map((item) => ListTile(dense: true, title: Text(item.label), trailing: Text('${item.count}'))),
    ])));
  }
}
