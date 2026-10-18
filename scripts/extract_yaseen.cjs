// Safe RAR extraction: args = <rar_dir> <out_dir>. Writes only .wav files, flat, basename only.
const fs = require('fs');
const path = require('path');
const { createExtractorFromData } = require('node-unrar-js');

(async () => {
  const [rarDir, outDir] = process.argv.slice(2);
  const rars = fs.readdirSync(rarDir).filter(f => f.endsWith('.rar') && !f.startsWith('Code'));
  for (const rar of rars) {
    const cls = rar.split('_')[0];              // AS, MR, MS, MVP, N
    const dest = path.join(outDir, cls);
    fs.mkdirSync(dest, { recursive: true });
    const buf = fs.readFileSync(path.join(rarDir, rar));
    const data = Uint8Array.from(buf).buffer;
    const extractor = await createExtractorFromData({ data });
    const { files } = extractor.extract();
    let n = 0, skipped = 0;
    for (const f of files) {
      const h = f.fileHeader;
      if (h.flags.directory) continue;
      const base = path.basename(h.name.replace(/\\/g, '/'));
      if (!/^[\w\-. ]+\.wav$/i.test(base) && !/\.wav$/i.test(base)) { skipped++; continue; }
      const safe = base.replace(/[^\w\-. ]/g, '_');
      const payload = (f.extraction && f.extraction.data) ? f.extraction.data : f.extraction;
      if (!payload) { skipped++; continue; }
      fs.writeFileSync(path.join(dest, safe), Buffer.from(payload));
      n++;
    }
    console.log(`${rar} -> ${cls}: ${n} wav written, ${skipped} skipped`);
  }
})().catch(e => { console.error('ERR', e); process.exit(1); });
