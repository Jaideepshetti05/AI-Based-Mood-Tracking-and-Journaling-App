import 'dart:convert';

import 'package:http/http.dart' as http;

import 'models.dart';

class ApiClient {
  ApiClient({String? baseUrl})
      : baseUrl = (baseUrl ?? const String.fromEnvironment(
          'API_BASE_URL',
          defaultValue: 'http://10.0.2.2:8000',
        )).replaceAll(RegExp(r'/$'), '');

  final String baseUrl;

  Future<JournalEntry> createEntry({
    required String text,
    required String mood,
    String? note,
    bool analyze = true,
  }) async {
    final response = await http.post(
      Uri.parse('$baseUrl/api/entries'),
      headers: _headersWithJson,
      body: jsonEncode({'text': text, 'mood': mood, 'note': note, 'analyze': analyze}),
    );
    _check(response);
    return JournalEntry.fromJson(jsonDecode(response.body) as Map<String, dynamic>);
  }

  Future<List<JournalEntry>> getEntries({String? mood, String? search}) async {
    final query = <String, String>{if (mood != null) 'mood': mood};
    final response = await http.get(
      Uri.parse('$baseUrl/api/entries').replace(queryParameters: query),
      headers: _headers,
    );
    _check(response);
    return (jsonDecode(response.body) as List<dynamic>)
        .map((item) => JournalEntry.fromJson(item as Map<String, dynamic>))
        .where((entry) => search == null || entry.text.toLowerCase().contains(search.toLowerCase()))
        .toList();
  }

  Future<Insights> getInsights({bool monthly = false}) async {
    final response = await http.get(
      Uri.parse('$baseUrl/api/insights/${monthly ? 'monthly' : 'weekly'}'),
      headers: _headers,
    );
    _check(response);
    return Insights.fromJson(jsonDecode(response.body) as Map<String, dynamic>);
  }

  Future<void> correctEntry(int id, {required bool helpful, String? emotion}) async {
    final response = await http.post(
      Uri.parse('$baseUrl/api/entries/$id/correction'),
      headers: _headersWithJson,
      body: jsonEncode({'helpful': helpful, 'corrected_emotion': emotion}),
    );
    _check(response);
  }

  Future<void> updateEntry(int id, {required String text, required String mood}) async {
    final response = await http.put(
      Uri.parse('$baseUrl/api/entries/$id'),
      headers: _headersWithJson,
      body: jsonEncode({'text': text, 'mood': mood, 'analyze': true}),
    );
    _check(response);
  }

  Future<void> deleteEntry(int id) async {
    final response = await http.delete(Uri.parse('$baseUrl/api/entries/$id'), headers: _headers);
    _check(response);
  }

  Future<String> exportData() async {
    final response = await http.get(Uri.parse('$baseUrl/api/export'), headers: _headers);
    _check(response);
    return response.body;
  }

  Future<void> deleteAllData() async {
    final response = await http.delete(Uri.parse('$baseUrl/api/data'), headers: _headers);
    _check(response);
  }

  Map<String, String> get _headers => {'X-User-ID': 'local-user'};
  Map<String, String> get _headersWithJson => {..._headers, 'Content-Type': 'application/json'};

  void _check(http.Response response) {
    if (response.statusCode < 200 || response.statusCode >= 300) {
      throw Exception('Request failed (${response.statusCode})');
    }
  }
}
