let b = `[Enter here]`.matchAll(/(\ Mariya\ Anti\ Paase\:)([\s\S]*?)(\[\d\d\:\d\d\ (am|pm), \d\d\/\d\d\/\d\d\d\d\]|$)/g);
let c = b.next();
let d = [];
let i = 1;
while (!c.done) {
  d.push(i++ + ': \n```\n' + c.value[2].trim() + '\n```');

  c = b.next();
}

copy('Generate: \n\n' + d.join(',\n\n\n\n'));
