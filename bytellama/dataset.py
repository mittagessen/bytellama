import random
import pathlib

import click

def chunk_text(text, max_bytes=2048):
    """
    Samples random chunks of max_bytes bytes from text, starting at whitespace
    positions. The number of samples is chosen so that the corpus is covered
    once on average (total_bytes / max_bytes).
    """
    text_bytes = text.encode('utf-8')
    total = len(text_bytes)
    if total == 0:
        return []

    # find all whitespace positions (valid chunk start points)
    ws_positions = [i for i, b in enumerate(text_bytes) if b in b' \t\n\r']
    if not ws_positions:
        return []

    n_samples = max(1, total // max_bytes)
    chunks = []
    for _ in range(n_samples):
        start = random.choice(ws_positions) + 1
        end = min(start + max_bytes, total)
        # back off to last whitespace to avoid splitting a word/character
        while end > start and end < total and text_bytes[end] not in b' \t\n\r':
            end -= 1
        if end <= start:
            continue
        chunk = text_bytes[start:end].decode('utf-8', errors='ignore').strip()
        if chunk:
            chunks.append(chunk)

    return chunks


@click.command()
@click.argument('manifest', type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path))
@click.option('-o', '--output', required=True, type=click.Path(path_type=pathlib.Path), help='Output directory for the dataset.')
@click.option('-c', '--chunk-size', default=2048, show_default=True, help='Maximum chunk size in bytes.')
@click.option('-s', '--seed', default=None, type=int, help='Random seed for reproducibility.')
def cli(manifest, output, chunk_size, seed):
    """
    Prepares a byte-chunked pretraining dataset from a manifest file.

    MANIFEST is a text file containing one input file path per line. Each
    input file is read, split into chunks of up to CHUNK_SIZE bytes at
    whitespace boundaries, and saved as a local Hugging Face dataset.
    """
    from datasets import Dataset
    from rich.progress import Progress

    if seed is not None:
        random.seed(seed)

    files = [pathlib.Path(line) for line in manifest.read_text(encoding='utf-8').splitlines() if line.strip()]

    all_chunks = []
    with Progress() as progress:
        task = progress.add_task('Chunking files', total=len(files))
        for path in files:
            text = path.read_text(encoding='utf-8')
            chunks = chunk_text(text, max_bytes=chunk_size)
            all_chunks.extend(chunks)
            progress.advance(task)

    random.shuffle(all_chunks)
    click.echo(f'Created {len(all_chunks)} chunks from {len(files)} file(s)')

    ds = Dataset.from_dict({'text': all_chunks})
    ds.save_to_disk(output)
    click.echo(f'Dataset saved to {output}')
