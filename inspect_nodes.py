# inspect_nodes.py
"""
Inspect node counts per document/collection.
Run: python inspect_nodes.py
"""
import json
from pathlib import Path
from collections import defaultdict
from config import settings
from src.storage_manager import StorageManager


def inspect_all():
    storage = StorageManager()
    collections = storage.list_collections()

    if not collections:
        print("No collections found. Run process_pdfs.py first.")
        return

    print("\n" + "="*60)
    print("NODE INSPECTION REPORT")
    print("="*60)

    for collection_name in collections:
        print(f"\n📦 Collection: {collection_name}")
        print("-" * 50)

        # ── ChromaDB counts (leaf/indexed nodes only) ──
        chroma_info = storage.get_collection_info(collection_name)
        print(f"  ChromaDB (indexed leaf nodes): {chroma_info['count']}")

        # ── Docstore counts (all nodes) ──
        docstore_file = storage.get_docstore_path(collection_name)

        if not docstore_file.exists():
            print(f"  Docstore: NOT FOUND")
            continue

        with open(docstore_file, 'r') as f:
            docs = json.load(f)

        total = len(docs)

        # Count by relationship type to identify parent/leaf
        # NodeRelationship: 4=PARENT, 5=CHILD
        has_child  = 0  # parent nodes (they have children)
        has_parent = 0  # leaf/mid nodes (they have a parent)
        leaf_nodes = 0  # true leaves (have parent, no children)
        
        page_counts = defaultdict(int)
        node_types  = defaultdict(int)

        for node_id, node_data in docs.items():
            rels = node_data.get("relationships", {})
            is_parent_node = "5" in rels  # has CHILD relationship
            is_child_node  = "4" in rels  # has PARENT relationship

            if is_parent_node:
                has_child += 1
                node_types["parent"] += 1
            elif is_child_node:
                leaf_nodes += 1
                node_types["leaf"] += 1
            else:
                node_types["root"] += 1

            # Count nodes per page
            page = node_data.get("metadata", {}).get("page")
            if page:
                page_counts[int(page)] += 1

        print(f"  Docstore total nodes:          {total}")
        print(f"    ├── Parent nodes :           {node_types['parent']}")
        print(f"    ├── Leaf nodes :             {node_types['leaf']}")
        print(f"    └── Root/other:              {node_types['root']}")

        # ── Per-page breakdown ──
        # if page_counts:
        #     print(f"\n  Per-page node counts:")
        #     print(f"  {'Page':<8} {'Nodes':<8} {'Bar'}")
        #     print(f"  {'-'*35}")
        #     for page in sorted(page_counts.keys()):
        #         count = page_counts[page]
        #         bar = "█" * count
        #         print(f"  {page:<8} {count:<8} {bar}")

        print(f"\n  Total pages covered: {len(page_counts)}")
        if page_counts:
            avg = total / len(page_counts)
            print(f"  Avg nodes per page:  {avg:.1f}")


if __name__ == "__main__":
    inspect_all()