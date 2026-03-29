import torch
import random


def inject_adversarial_edges(data, num_fake_nodes=1000, num_edges_per_fake=3):

    edge_index = data.edge_index.clone()
    num_nodes = data.num_nodes

    new_edges = []

    # =========================
    # Create fake nodes + edges
    # =========================
    for i in range(num_fake_nodes):

        fake_node = num_nodes + i

        for _ in range(num_edges_per_fake):
            target = random.randint(0, num_nodes - 1)

            # bidirectional edges
            new_edges.append([fake_node, target])
            new_edges.append([target, fake_node])

    new_edges = torch.tensor(new_edges).t().contiguous()

    # =========================
    # Update edge_index
    # =========================
    edge_index = torch.cat([edge_index, new_edges], dim=1)

    # =========================
    # Update node features
    # =========================
    fake_features = torch.zeros((num_fake_nodes, data.x.size(1)))
    data.x = torch.cat([data.x, fake_features], dim=0)

    # =========================
    # Update labels
    # -1 → fake nodes (ignored in training)
    # =========================
    fake_labels = torch.full((num_fake_nodes,), -1)
    data.y = torch.cat([data.y, fake_labels], dim=0)

    # =========================
    # Update time_steps (IMPORTANT)
    # =========================
    fake_time_steps = torch.zeros(num_fake_nodes, dtype=data.time_steps.dtype)
    data.time_steps = torch.cat([data.time_steps, fake_time_steps], dim=0)

    # =========================
    # Update masks (CRITICAL FIX)
    # =========================
    fake_train_mask = torch.zeros(num_fake_nodes, dtype=data.train_mask.dtype)
    fake_test_mask = torch.zeros(num_fake_nodes, dtype=data.test_mask.dtype)

    data.train_mask = torch.cat([data.train_mask, fake_train_mask], dim=0)
    data.test_mask = torch.cat([data.test_mask, fake_test_mask], dim=0)

    # =========================
    # Assign updated edge_index
    # =========================
    data.edge_index = edge_index

    return data