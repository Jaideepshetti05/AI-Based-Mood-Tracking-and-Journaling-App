class Analysis {
  const Analysis({
    required this.sentiment,
    required this.compoundScore,
    required this.emotion,
    required this.modelName,
    this.confidence,
    required this.limitation,
  });

  final String sentiment;
  final double compoundScore;
  final String emotion;
  final String modelName;
  final double? confidence;
  final String limitation;

  factory Analysis.fromJson(Map<String, dynamic> json) => Analysis(
        sentiment: json['sentiment'] as String,
        compoundScore: (json['compound_score'] as num).toDouble(),
        emotion: json['emotion'] as String,
        modelName: json['model_name'] as String,
        confidence: (json['confidence'] as num?)?.toDouble(),
        limitation: json['limitation'] as String,
      );
}

class Correction {
  const Correction({required this.helpful, this.correctedEmotion});

  final bool helpful;
  final String? correctedEmotion;

  factory Correction.fromJson(Map<String, dynamic> json) => Correction(
        helpful: json['helpful'] as bool,
        correctedEmotion: json['corrected_emotion'] as String?,
      );
}

class JournalEntry {
  const JournalEntry({
    required this.id,
    required this.text,
    required this.mood,
    required this.createdAt,
    this.note,
    this.analysis,
    this.correction,
  });

  final int id;
  final String text;
  final String mood;
  final String? note;
  final DateTime createdAt;
  final Analysis? analysis;
  final Correction? correction;

  factory JournalEntry.fromJson(Map<String, dynamic> json) => JournalEntry(
        id: json['id'] as int,
        text: json['text'] as String,
        mood: json['mood'] as String,
        note: json['note'] as String?,
        createdAt: DateTime.parse(json['created_at'] as String),
        analysis: json['analysis'] == null ? null : Analysis.fromJson(json['analysis'] as Map<String, dynamic>),
        correction: json['correction'] == null ? null : Correction.fromJson(json['correction'] as Map<String, dynamic>),
      );
}

class Distribution {
  const Distribution(this.label, this.count);
  final String label;
  final int count;

  factory Distribution.fromJson(Map<String, dynamic> json) => Distribution(json['label'] as String, json['count'] as int);
}

class Insights {
  const Insights({
    required this.entryCount,
    required this.mostFrequentMood,
    required this.moodDistribution,
    required this.sentimentDistribution,
    required this.emotionDistribution,
    required this.agreementRate,
  });

  final int entryCount;
  final String? mostFrequentMood;
  final List<Distribution> moodDistribution;
  final List<Distribution> sentimentDistribution;
  final List<Distribution> emotionDistribution;
  final double? agreementRate;

  factory Insights.fromJson(Map<String, dynamic> json) => Insights(
        entryCount: json['entry_count'] as int,
        mostFrequentMood: json['most_frequent_mood'] as String?,
        moodDistribution: _distributions(json['mood_distribution']),
        sentimentDistribution: _distributions(json['sentiment_distribution']),
        emotionDistribution: _distributions(json['emotion_distribution']),
        agreementRate: (json['agreement_rate'] as num?)?.toDouble(),
      );

  static List<Distribution> _distributions(dynamic value) =>
      (value as List<dynamic>).map((item) => Distribution.fromJson(item as Map<String, dynamic>)).toList();
}
