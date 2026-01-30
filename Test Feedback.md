# Testbericht

## Überblick
Dieser Bericht dokumentiert alle Tests im Repository, beschreibt was sie prüfen, benennt Schwachstellen der Tests selbst (nicht der Produktivlogik) und schlägt konkrete Verbesserungen vor. Der fehlgeschlagene Test `test_openai_tool_error_sanitization` wird dabei ausdrücklich eingeordnet.

## tests/test_gemini_core.py

### test_generic_gemini_initialization
**Zweck:** Prüft, dass `GenericGemini` korrekt initialisiert wird (Model-Name, Client, System-Instruktion).【F:tests/test_gemini_core.py†L8-L17】

**Schwachstellen:**
- Validiert nur einfache Attribute, keine Interaktion mit dem Client oder Randfälle (z. B. fehlende/ungültige Konfiguration).【F:tests/test_gemini_core.py†L8-L17】

**Verbesserungsvorschläge:**
- Negativtests für fehlende oder ungültige Parameter hinzufügen (z. B. `model_name=None`).
- Überprüfen, ob optionale Konfigurationen (Timeouts, Registry) korrekt gesetzt werden.

### test_ask_method
**Zweck:** Prüft, dass `ask()` einen `GeminiMessageResponse` liefert, den Text korrekt setzt, Tokens ausliest und `send_message` aufruft.【F:tests/test_gemini_core.py†L19-L48】

**Schwachstellen:**
- Starker Einsatz von `MagicMock` ohne Validierung der realen API-Struktur (kann echte SDK-Fehler verdecken).【F:tests/test_gemini_core.py†L19-L48】
- Prüft keine Fehlerszenarien oder leere Antworten.

**Verbesserungsvorschläge:**
- Fehlerszenarien (Exception aus `send_message`) testen.
- Zusätzliche Assertions für `usage_metadata=None` und leere `parts`.

### test_chat_method
**Zweck:** Prüft, dass `chat()` eine Chat-Antwort mit History liefert und den letzten Text korrekt übernimmt.【F:tests/test_gemini_core.py†L51-L79】

**Schwachstellen:**
- Testet nur den Minimalfall (eine History-Entry, kein Tool-Calling).
- Überprüft nicht, ob History-Inhalte semantisch korrekt sind (z. B. Rollen/Parts).【F:tests/test_gemini_core.py†L51-L79】

**Verbesserungsvorschläge:**
- Validieren, dass History korrekt aus SDK-Objekten gemappt wird.
- Zusatztests mit mehreren History-Einträgen.

### test_function_calling
**Zweck:** Simuliert Tool-Calling und prüft, dass das Tool ausgeführt und eine zweite `send_message`-Runde gemacht wird.【F:tests/test_gemini_core.py†L81-L135】

**Schwachstellen:**
- Tool-Definition wird manuell „hineinmocked“, um Pydantic zu umgehen – dadurch werden echte Registrierungsfehler nicht erfasst.【F:tests/test_gemini_core.py†L89-L106】
- Keine Überprüfung des gesendeten Tool-Results an das Modell.

**Verbesserungsvorschläge:**
- Einen echten `@registry.tool`-Pfad testen (statt manuellem Mocking).
- Assertions zum Inhalt der Tool-Response an `send_message` hinzufügen.

### test_function_calling_with_empty_args
**Zweck:** Prüft, dass Tools mit `None`-Argumenten korrekt ohne Parameter aufgerufen werden.【F:tests/test_gemini_core.py†L137-L186】

**Schwachstellen:**
- Wieder manueller Tool-Mock statt realer Registry-Registrierung.
- Keine Prüfung, dass Tool-Result an das Modell korrekt serialisiert wird.

**Verbesserungsvorschläge:**
- Serialisierungs- und Response-Assertions ergänzen.
- Varianten testen: leere Dicts, leere JSON-Strings.

## tests/test_openai_core.py

### test_generic_openai_initialization
**Zweck:** Prüft einfache Initialisierungswerte für `GenericOpenAI`.【F:tests/test_openai_core.py†L19-L29】

**Schwachstellen:**
- Keine Validierung von Default-Config-Parametern oder fehlerhaften Parametern.

**Verbesserungsvorschläge:**
- Tests für Standardwerte und Fehlkonfigurationen ergänzen.

### test_ask_method
**Zweck:** Prüft `ask()` für OpenAI: Response-Typ, Text, Tokenzählung und die in-place modifizierte `messages`-Liste.【F:tests/test_openai_core.py†L31-L77】

**Schwachstellen:**
- Abhängigkeit von in-place Änderung der `messages`-Liste macht den Test empfindlich gegen Refactorings.【F:tests/test_openai_core.py†L60-L77】
- Keine Fehlerpfade oder Tool-Calling berücksichtigt.

**Verbesserungsvorschläge:**
- Erwartete `messages` als separate Kopie prüfen, um Implementation-Details zu entkoppeln.
- Fehlerfälle (API-Exception) testen.

### test_chat_method
**Zweck:** Prüft, dass `chat()` eine Antwort liefert und die History Rollenfolge korrekt ist.【F:tests/test_openai_core.py†L78-L113】

**Schwachstellen:**
- Annahme, dass die History exakt 3 Einträge hat, ist stark an die interne Implementierung gebunden.【F:tests/test_openai_core.py†L103-L111】

**Verbesserungsvorschläge:**
- Überprüfung auf „enthält System/User/Assistant“ statt fixer Länge.
- Zusätzliche Tests für vorhandene History und Tool-Calling.

### test_function_calling
**Zweck:** Simuliert Tool-Calling über OpenAI und prüft Tool-Execution sowie korrekte Tool-Messages.【F:tests/test_openai_core.py†L115-L199】

**Schwachstellen:**
- Tool-Definition wird manuell in die Registry geschrieben (kein Real-Pfad).【F:tests/test_openai_core.py†L120-L133】
- Assertions hängen an der konkreten Reihenfolge/Anzahl der `messages` inklusive in-place Mutation.【F:tests/test_openai_core.py†L173-L199】

**Verbesserungsvorschläge:**
- Echten `@registry.tool`-Pfad testen.
- Reihenfolge-Assertions lockern und stattdessen auf inhaltliche Elemente prüfen.

## tests/test_registry.py

### test_registry_tool_decorator
**Zweck:** Prüft, dass der Decorator Tools registriert und Docstring/Logik funktioniert.【F:tests/test_registry.py†L14-L28】

**Schwachstellen:**
- Kein Test für Typvalidierung bei komplexen Signaturen.

**Verbesserungsvorschläge:**
- Mehr Varianten testen (Default-Parameter, Optional, Union).

### test_gemini_registry_tool_object
**Zweck:** Prüft, dass Gemini-Tool-Objekt korrekt erstellt wird.【F:tests/test_registry.py†L30-L44】

**Schwachstellen:**
- Prüft nicht alle Felder (Parameter-Schema wird nicht validiert).【F:tests/test_registry.py†L30-L44】

**Verbesserungsvorschläge:**
- Assertions zu Parametern/Schema hinzufügen.

### test_openai_registry_tool_object
**Zweck:** Prüft OpenAI-Tool-Objekt-Struktur und Schema-Grundlagen.【F:tests/test_registry.py†L45-L63】

**Schwachstellen:**
- Minimaler Schema-Check, keine Prüfung der `properties`-Details.

**Verbesserungsvorschläge:**
- Schema-Felder detaillierter prüfen.

### test_registry_missing_docstring
**Zweck:** Stellt sicher, dass fehlende Docstrings zu `ToolValidationError` führen.【F:tests/test_registry.py†L64-L70】

**Schwachstellen:**
- Erwartet konkrete Fehlermeldung (Match-String), was bei Textänderungen bricht.【F:tests/test_registry.py†L64-L70】

**Verbesserungsvorschläge:**
- Auf Error-Typ und ggf. Error-Code prüfen, nicht auf exakten Text.

### test_registry_missing_param_description
**Zweck:** Prüft Fehler bei fehlenden Parameter-Beschreibungen.【F:tests/test_registry.py†L71-L78】

**Schwachstellen:**
- Wie oben: empfindlich gegenüber Textänderungen der Fehlermeldung.

**Verbesserungsvorschläge:**
- Fehlerklasse + kurzer Match auf stabile Schlüsselwörter nutzen.

### test_nested_pydantic_models_schema_resolution
**Zweck:** Prüft, dass verschachtelte Pydantic-Schemata ohne `$ref` aufgelöst werden.【F:tests/test_registry.py†L79-L126】

**Schwachstellen:**
- Überprüft keine Sonderfälle (z. B. `Optional`, `Union`, rekursive Modelle).
- Starker Fokus auf JSON-Struktur anstatt semantische Validierung.

**Verbesserungsvorschläge:**
- Zusätzliche Modellvarianten ergänzen (Listen, Optional, Union, Selbst-Referenz).

### test_tool_without_parameters
**Zweck:** Verifiziert Tools ohne Parameter und leeres Schema-Objekt.【F:tests/test_registry.py†L128-L151】

**Schwachstellen:**
- Prüft nur erfolgreiche Registrierung, keine Fehlerfälle für `None`-Schema.

**Verbesserungsvorschläge:**
- Tests für explizit verbotene Parameterstrukturen ergänzen.

## tests/test_tool_helper.py

### test_tool_helper_initialization
**Zweck:** Prüft, dass `ToolHelper` korrekt initialisiert wird.【F:tests/test_tool_helper.py†L24-L40】

**Schwachstellen:**
- Keine Validierung von Default-Werten oder fehlerhaften Parametern.

**Verbesserungsvorschläge:**
- Negativtests und Defaultwerte prüfen.

### test_handle_function_calls_no_calls
**Zweck:** Prüft, dass ohne Tool-Calls keine zusätzliche API-Interaktion stattfindet.【F:tests/test_tool_helper.py†L42-L72】

**Schwachstellen:**
- Setzt `messages`-Länge hart voraus und nutzt Mocks ohne SDK-Validierung.

**Verbesserungsvorschläge:**
- Assertions auf semantische Inhalte statt exakter Länge.

### test_handle_function_calls_execution
**Zweck:** Simuliert Tool-Execution und prüft, dass Tool-Result korrekt als Tool-Message eingetragen wird.【F:tests/test_tool_helper.py†L74-L143】

**Schwachstellen:**
- Mocking der Tool-Definition umgeht reale Registry-Validierung.【F:tests/test_tool_helper.py†L79-L84】
- Testet nur eine Tool-Call-Iteration (keine Mehrfach-Calls).

**Verbesserungsvorschläge:**
- Mehrfache Tool-Calls in einer Antwort testen.
- Echte Registry-Registrierung verwenden.

### test_handle_function_calls_error
**Zweck:** Prüft, dass Tool-Exceptions zu einer sanitisierten Fehlermeldung werden.【F:tests/test_tool_helper.py†L145-L208】

**Schwachstellen:**
- Erwartet exakten Fehlertext (`An internal error occurred during tool execution.`).【F:tests/test_tool_helper.py†L202-L208】

**Verbesserungsvorschläge:**
- Fehlertext weniger strikt prüfen (z. B. enthält „internal error“).

### test_handle_function_calls_empty_arguments
**Zweck:** Prüft, dass leere Argument-Strings Tools ohne Parameter ausführen.【F:tests/test_tool_helper.py†L209-L262】

**Schwachstellen:**
- Kein Test für leere JSON-Objekte (`{}`) oder `None`.

**Verbesserungsvorschläge:**
- Varianten für `None`, `{}`, und Whitespaces ergänzen.

### test_handle_function_calls_invalid_arguments
**Zweck:** Prüft, dass ungültige Argumentformate zu Fehlern in Tool-Responses führen.【F:tests/test_tool_helper.py†L264-L334】

**Schwachstellen:**
- Prüft nur das Vorhandensein von „error“, nicht den Inhalt oder Typ der Fehlermeldung.

**Verbesserungsvorschläge:**
- Fehler-Details (z. B. „invalid arguments“) zusätzlich prüfen.

## tests/test_security_fixes.py

### test_gemini_tool_error_sanitization
**Zweck:** Prüft, dass Gemini-Tool-Exceptions geloggt und für das Modell sanitisiert werden.【F:tests/test_security_fixes.py†L12-L63】

**Schwachstellen:**
- Starke Abhängigkeit von internem Logging-Text („Unexpected error executing tool ...“).【F:tests/test_security_fixes.py†L52-L55】
- Mocking des SDK kann echte Serialisierungsprobleme verstecken.

**Verbesserungsvorschläge:**
- Logging-Assertion auf Error-Level oder structured logs reduzieren.
- Zusätzliche Tests für verschiedene Exception-Typen.

### test_gemini_tool_timeout
**Zweck:** Prüft, dass Tool-Timeouts korrekt als Fehler zurückgegeben werden.【F:tests/test_security_fixes.py†L77-L122】

**Schwachstellen:**
- Nutzt echtes `asyncio.sleep(2)` mit kurzem Timeout (Test kann langsam/flaky sein).【F:tests/test_security_fixes.py†L96-L101】

**Verbesserungsvorschläge:**
- Zeit mittels `asyncio.wait_for`/Fake-Timer oder `pytest`-Monkeypatch beschleunigen.

### test_schema_recursion_limit
**Zweck:** Prüft, dass `_resolve_schema_refs` bei Rekursion abbricht.【F:tests/test_security_fixes.py†L124-L166】

**Schwachstellen:**
- Testet eine interne Methode, die nicht Teil der öffentlichen API ist (fragiler Test).【F:tests/test_security_fixes.py†L146-L166】

**Verbesserungsvorschläge:**
- Über publice Tool-Registrierung testen (Input -> RecursionError) statt private Methode.

### test_default_config_values
**Zweck:** Prüft Standardwerte in `LLMConfig` (Tokens/Temperature).【F:tests/test_security_fixes.py†L168-L174】

**Schwachstellen:**
- Test bricht bei legitimen Default-Änderungen (schwierige Wartbarkeit).【F:tests/test_security_fixes.py†L168-L174】

**Verbesserungsvorschläge:**
- Defaults in Konstante definieren und dort referenzieren oder per Snapshot dokumentieren.

### test_openai_tool_error_sanitization (fehlgeschlagen)
**Zweck:** Prüft, dass OpenAI-Tool-Exceptions geloggt und als sanitized Tool-Message zurückgegeben werden.【F:tests/test_security_fixes.py†L177-L266】

**Schwachstellen (testseitig):**
- Verwendet `GeminiToolRegistry` statt `OpenAIToolRegistry`, was zu inkonsistenter Tool-Definition führen kann (mögliche Fehlerquelle).【F:tests/test_security_fixes.py†L221-L228】
- Erwartet konkreten Fehlertext „An internal error occurred“, ohne Spielraum für alternative Sanitization-Formate.【F:tests/test_security_fixes.py†L262-L266】
- Starker Mocking-Fokus, ohne Validierung eines realistischen OpenAI-SDK-Objekts.

**Verbesserungsvorschläge:**
- OpenAI-Registry verwenden und Tool-Registration realistisch durchführen.
- Fehlertext weniger strikt prüfen (z. B. `"internal error" in content`).
- Ein Test für logging und ein Test für Tool-Response trennen (Single-Responsibility).

## tests/test_weaknesses.py

### test_duplicate_tool_registration_overwrite
**Zweck:** Prüft, dass doppelte Tool-Namen einen Fehler auslösen (keine Über­schreibung).【F:tests/test_weaknesses.py†L10-L32】

**Schwachstellen:**
- Der Test beschreibt „Weakness“, prüft aber das gewünschte Fehlverhalten; wenn das Verhalten geändert wird, bleibt der Test missverständlich benannt.【F:tests/test_weaknesses.py†L10-L32】

**Verbesserungsvorschläge:**
- Testnamen und Kommentare aktualisieren, wenn Verhalten behoben wurde.

### test_circular_pydantic_models_recursion_error
**Zweck:** Prüft, dass zirkuläre Pydantic-Modelle nicht zu RecursionError führen (fehlschlagen soll).【F:tests/test_weaknesses.py†L34-L67】

**Schwachstellen:**
- Test schlägt fehl, wenn irgendeine Exception kommt, aber akzeptiert kein alternatives Fehlerbild (z. B. eigener Error-Typ).【F:tests/test_weaknesses.py†L54-L67】

**Verbesserungsvorschläge:**
- Konkrete erwartete Exception und Fehlermeldung definieren, wenn die Logik verbessert wird.

### test_max_function_loops_exceeded
**Zweck:** Zeigt, dass ein Loop-Limit zu einer leeren Antwort führen kann (Potential-Schwäche).【F:tests/test_weaknesses.py†L70-L115】

**Schwachstellen:**
- Der Test misst nur den leeren Text, prüft nicht, ob das Limit korrekt gezählt wurde.【F:tests/test_weaknesses.py†L102-L114】

**Verbesserungsvorschläge:**
- Prüfen, dass tatsächlich `max_function_loops` Durchläufe stattfanden.
- Erwartete „fallback“-Meldung testen, wenn implementiert.

### test_non_serializable_tool_return_crash
**Zweck:** Prüft, dass nicht serialisierbare Tool-Returns eine Ausnahme auslösen (zeigt Crash-Pfad).【F:tests/test_weaknesses.py†L117-L175】

**Schwachstellen:**
- Test hängt stark von Mock-Implementierung ab; echte SDK-Serialisierungsfehler können anders aussehen.【F:tests/test_weaknesses.py†L132-L166】

**Verbesserungsvorschläge:**
- Separaten Unit-Test für Serialisierung (ohne SDK) hinzufügen.
- Prüfen, dass Fehler intern abgefangen und sanitisiert wird (wenn fix geplant).

### test_empty_prompt_handling
**Zweck:** Prüft, dass leere Prompts eine Exception auslösen (zeigt fehlende Validierung).【F:tests/test_weaknesses.py†L177-L208】

**Schwachstellen:**
- Erwartet eine generische Exception; kein spezifischer Fehler-Typ, keine genaue Meldungsklasse.【F:tests/test_weaknesses.py†L191-L208】

**Verbesserungsvorschläge:**
- Eigene `ValidationError`-Klasse prüfen, wenn eingeführt.

## tests/test_weaknesses_2.py

### test_arg_type_coercion_failure
**Zweck:** Zeigt, dass Typ-Coercion für Tool-Args nicht erfolgt (String statt Int).【F:tests/test_weaknesses_2.py†L12-L65】

**Schwachstellen:**
- Erwartet explizit „30“ als Ergebnis, was eher ein Soll-Verhalten als Schwachstellen-Beweis ist; der Test schlägt fehl, sobald Type-Coercion implementiert wird (dann ist der „Weakness“-Test obsolet).【F:tests/test_weaknesses_2.py†L55-L65】

**Verbesserungsvorschläge:**
- Test umbenennen oder in „Regressionstest für Fix“ umwandeln, sobald Coercion implementiert ist.

### test_var_args_registration_failure
**Zweck:** Prüft, dass Tools mit `*args` aktuell nicht registriert werden können.【F:tests/test_weaknesses_2.py†L68-L90】

**Schwachstellen:**
- Abhängig vom exakten Fehltext (fragil).【F:tests/test_weaknesses_2.py†L84-L90】

**Verbesserungsvorschläge:**
- Error-Klasse plus Match auf stabilen Begriff prüfen.

### test_ask_hides_tool_execution_details
**Zweck:** Zeigt, dass `ask()` keine Tool-Execution-History zurückgibt.【F:tests/test_weaknesses_2.py†L92-L133】

**Schwachstellen:**
- Prüft nur Attribute (`hasattr`), nicht ob eine alternative API existiert, um Tool-History einzusehen.

**Verbesserungsvorschläge:**
- Falls ein neues Feld eingeführt wird, Test entsprechend anpassen (z. B. `response.history`).

### test_nested_schema_title_leak
**Zweck:** Prüft, dass `title` im verschachtelten Schema noch vorhanden ist (Weakness).【F:tests/test_weaknesses_2.py†L135-L168】

**Schwachstellen:**
- Der Test erwartet explizit „title“ im nested Schema; wenn Bug gefixt wird, muss der Test entfernt/angepasst werden.【F:tests/test_weaknesses_2.py†L155-L167】

**Verbesserungsvorschläge:**
- Bei Fix: Test in Regressionstest umwandeln, der sicherstellt, dass `title` **nicht** vorhanden ist.

### test_sync_tool_blocks_event_loop
**Zweck:** Versucht zu zeigen, dass synchrone Tools den Event-Loop blockieren.【F:tests/test_weaknesses_2.py†L169-L259】

**Schwachstellen:**
- Die Assertion am Ende ist invertiert: `assert task_ran_time < start_time + 0.1` erwartet ein *schnelles* Task-Run, obwohl der Test das Blockieren demonstrieren will.【F:tests/test_weaknesses_2.py†L255-L259】
- Timing-basierte Tests sind anfällig für Flakiness in CI.

**Verbesserungsvorschläge:**
- Assertion korrigieren (z. B. `>= start_time + 0.2`), oder den Test durch deterministisches Event-Loop-Instrumenting ersetzen.
- Alternativ: Tool-Ausführung via `run_in_executor` testbar machen.
