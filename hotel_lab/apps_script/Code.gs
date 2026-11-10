/**
 * Hotel Lab <-> Google Sheet bridge.
 *
 * SETUP (once, about 5 minutes):
 *  1. Open the Google Sheet  ->  Extensions  ->  Apps Script.
 *  2. Delete everything in Code.gs, paste this whole file, click Save.
 *  3. Deploy -> New deployment -> type: Web app.
 *       Execute as: Me        Who has access: Anyone
 *     Click Deploy, allow the permissions, and copy the "Web app URL" (it ends in /exec).
 *  4. In the Hotel Lab website (Data source panel) paste that URL and the api_token from the
 *     Settings tab, then press "Sync from Sheet".
 *
 * Security: the link only works with the api_token from the Settings tab. Do not share either.
 * If you change this code later: Deploy -> Manage deployments -> Edit (pencil) -> Version: New version -> Deploy.
 * The link stays the same, so nothing changes in the website.
 *
 * Version 2 (Nov 2026): the website may also write the Models tab (list of models, never API keys).
 */

var PROTECTED_READ = ['Results', 'Verdicts', 'Runs'];   // large tabs, not needed to run experiments
var WRITE_TABS = ['Results', 'Verdicts', 'Runs', 'Models'];   // the only tabs the website may change

function getToken_() {
  var sh = SpreadsheetApp.getActiveSpreadsheet().getSheetByName('Settings');
  if (!sh) return '';
  var values = sh.getDataRange().getDisplayValues();
  for (var i = 1; i < values.length; i++) {
    if (String(values[i][0]).trim() === 'api_token') return String(values[i][1]).trim();
  }
  return '';
}

function json_(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj)).setMimeType(ContentService.MimeType.JSON);
}

function checkToken_(token) {
  var expected = getToken_();
  return expected !== '' && String(token || '') === expected;
}

function doGet(e) {
  try {
    var p = (e && e.parameter) || {};
    if (!checkToken_(p.token)) return json_({ ok: false, error: 'bad or missing token (see Settings tab: api_token)' });
    var ss = SpreadsheetApp.getActiveSpreadsheet();
    var includeAll = p.all === '1';
    var tabs = {};
    ss.getSheets().forEach(function (sh) {
      var name = sh.getName();
      if (!includeAll && PROTECTED_READ.indexOf(name) >= 0) return;
      // display values = exactly what you see in the cell (no date/number reformatting surprises)
      tabs[name] = sh.getDataRange().getDisplayValues();
    });
    return json_({ ok: true, tabs: tabs, sheet: ss.getName(), read_at: new Date().toISOString() });
  } catch (err) {
    return json_({ ok: false, error: String(err) });
  }
}

function doPost(e) {
  var lock = LockService.getScriptLock();
  try {
    lock.waitLock(30000);
    var body = JSON.parse(e.postData.contents);
    if (!checkToken_(body.token)) return json_({ ok: false, error: 'bad or missing token' });
    if (body.action !== 'append' && body.action !== 'replace') return json_({ ok: false, error: 'unknown action ' + body.action });
    if (!body.tab || WRITE_TABS.indexOf(body.tab) < 0) {
      return json_({ ok: false, error: 'the website may only write the Results, Verdicts, Runs and Models tabs' });
    }
    var ss = SpreadsheetApp.getActiveSpreadsheet();
    var sh = ss.getSheetByName(body.tab) || ss.insertSheet(body.tab);
    var rows = body.rows || [];

    if (body.action === 'append') {
      if (rows.length === 0) return json_({ ok: true, appended: 0 });
      var header = sh.getLastRow() > 0
        ? sh.getRange(1, 1, 1, Math.max(sh.getLastColumn(), 1)).getDisplayValues()[0].map(String)
        : [];
      while (header.length && header[header.length - 1] === '') header.pop();   // keep positions of blank columns
      // add any new columns at the end
      rows.forEach(function (r) {
        Object.keys(r).forEach(function (k) { if (header.indexOf(k) < 0) header.push(k); });
      });
      sh.getRange(1, 1, 1, header.length).setValues([header]).setFontWeight('bold');
      var values = rows.map(function (r) {
        return header.map(function (k) { var v = r[k]; return v === null || v === undefined ? '' : String(v); });
      });
      var start = sh.getLastRow() + 1;
      var range = sh.getRange(start, 1, values.length, header.length);
      range.setNumberFormat('@');      // store as text: no auto-conversion of ids or dates
      range.setValues(values);
      return json_({ ok: true, appended: values.length });
    }

    if (body.action === 'replace') {
      var cols = body.columns || (rows.length ? Object.keys(rows[0]) : []);
      sh.clearContents();
      if (cols.length === 0) return json_({ ok: true, written: 0 });
      var out = [cols].concat(rows.map(function (r) {
        return cols.map(function (k) { var v = r[k]; return v === null || v === undefined ? '' : String(v); });
      }));
      var rg = sh.getRange(1, 1, out.length, cols.length);
      rg.setNumberFormat('@');
      rg.setValues(out);
      sh.getRange(1, 1, 1, cols.length).setFontWeight('bold');
      return json_({ ok: true, written: rows.length });
    }

    return json_({ ok: false, error: 'unknown action ' + body.action });
  } catch (err) {
    return json_({ ok: false, error: String(err) });
  } finally {
    lock.releaseLock();
  }
}
