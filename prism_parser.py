import json

from State import State


# class MyEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, bytes):
#             return str(obj, encoding='utf-8')
#         elif isinstance(obj, int):
#             return int(obj)
#         elif isinstance(obj, float):
#             return float(obj)
#         elif isinstance(obj, State):
#             return obj.to_dict()
#         # elif isinstance(obj, array):
#         #    return obj.tolist()
#         else:
#             return super(MyEncoder, self).default(obj)


class PrismParser:
    def __init__(self, K, env_graph, policy_graph):
        self.env_graph = env_graph
        self.policy_graph = policy_graph
        self.node_num = K + 2

        self.env_action_space = self._env_action_space()
        self.special_action_id = 0

    def _env_action_space(self):
        _set = set()
        for tag in self.env_graph.nodes:
            node = self.env_graph.nodes[tag]
            for child_tag, action in node.children:
                _set.add(action)
        return _set

    def add_transition(self, guard, signal='', updates=None):
        if updates is None:
            updates = []
        updates_str = ''
        # print(updates)
        for index, [weight, update] in enumerate(updates):
            if index == len(updates) - 1:
                updates_str += f'{weight} : {update}'
            else:
                updates_str += f'{weight}: {update} + '
        return f'[{signal}] {guard} -> {updates_str};'

    def env_transition(self, node, signal='update'):
        action_dict = {}  # action_id -> [[next_state_id], weight_sum]
        for tag, action_id in node.children:
            weight = node.children[(tag, action_id)]
            if action_id not in action_dict:
                action_dict[action_id] = [[tag], weight]
            else:
                action_dict[action_id][0].append(tag)
                action_dict[action_id][1] += weight

        # print(action_dict)

        codes = []
        for action_id in action_dict:
            next_state_ids = action_dict[action_id][0]
            weight_sum = action_dict[action_id][1]
            updates = []
            for next_state_id in next_state_ids:
                weight = node.children[(next_state_id, action_id)]
                updates.append([f'{weight}/{weight_sum}' if weight_sum != 0 else '', f'(state\'={next_state_id}) & (sched\'=0)'])
            codes.append(self.add_transition(f'(state={node.state.tag}) & (action={action_id}) & (sched=1)', updates=updates))
        return '\n'.join(codes)

    def policy_transition(self, node, signal='update'):
        action_dict = {}  # action_id -> weight
        weight_sum = 0
        for tag, action_id in node.children:
            weight = node.children[(tag, action_id)]
            if action_id not in self.env_action_space:
                action_id = self.special_action_id
            if action_id not in action_dict:
                action_dict[action_id] = weight
            else:
                action_dict[action_id] += weight
            weight_sum += weight

        updates = []
        for action_id in action_dict:
            weight = action_dict[action_id]
            updates.append([f'{weight}/{weight_sum}' if weight_sum != 0 else '', f'(action\'={action_id}) & (sched\'=1)'])
        return self.add_transition(f'(state={node.state.tag}) & (sched=0)', updates=updates)

    def declaration(self):
        d = f'''
mdp

global sched: [0..1] init 0;
'''
        return d

    def parse_env(self):
        t = ""
        for tag in self.env_graph.nodes:
            if tag == self.node_num - 1:
                continue
            t += self.env_transition(self.env_graph.nodes[tag], signal='update') + "\n"

        code = f'''
module Env
state : [0..{self.node_num - 1}] init 0;

[] (action={self.special_action_id}) & (sched=1) -> 1 : (sched\'=0);
{t}
endmodule
'''
        return code

    def parse_policy(self):
        t = ""
        for tag in self.policy_graph.nodes:
            if tag == self.node_num - 1:
                continue
            t += self.policy_transition(self.policy_graph.nodes[tag], signal='update') + "\n"

        code = f'''

module Policy
action: [0..{self.env_graph.action_spliter.action_len}];

{t}
endmodule
'''
        return code

    def parse(self, save=False, filename='code.prism'):
        code = f'{self.declaration()}\n{self.parse_policy()}\n{self.parse_env()}'
        if save:
            with open(filename, 'w') as f:
                f.write(code)
        return code
