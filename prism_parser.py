import json
import os

import base
import utils


class PrismParser:
    def __init__(self, K, env_graph, policy_graph, props_dict):
        self.env_graph = env_graph
        self.policy_graph = policy_graph
        self.node_num = K + 2
        self.props_dict = props_dict

        self.env_action_space = self._env_action_space()
        self.special_action_id = 0

    def _env_action_space(self) -> set:
        """
        环境的动作空间
        :return:
        """
        _set = set()
        for tag in self.env_graph.nodes:
            node = self.env_graph.nodes[tag]
            for child_tag, action in node.children:
                _set.add(action)
        return _set

    def add_transition(self, guard: str, signal: str = '', updates: list = None) -> str:
        """
        增加迁移
        :param guard: guard 条件
        :param signal: 信号
        :param updates: 更新操作列表
        :return:
        """
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

    def env_transition(self, node: base.Node, signal='update') -> str:
        """
        环境模型的迁移
        :param node: 当前节点
        :param signal: 目前忽略该参数
        :return:
        """
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
                updates.append([f'{weight}/{weight_sum}' if weight_sum != 0 else '', f'(state\'={next_state_id}) & (sched\'={0 if len(self.props_dict["name"]) == 0 else 2})'])
            codes.append(self.add_transition(f'(state={node.state.tag}) & (action={action_id}) & (sched=1)', updates=updates))
        return '\n'.join(codes)

    def policy_transition(self, node: base.Node, signal='update') -> str:
        """
        策略模型的迁移
        :param node: 当前节点
        :param signal: 目前忽略该参数
        :return:
        """
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

    def policy_reward(self, node):
        """
        不支持负奖励，虽然可以用两组 reward 分别计算正负奖励
        rewards "positive"
        ...
        endrewards

        rewards "negative"
        ...
        endrewards

        性质写为
        R{"positive"}=? [F C<={steps}]
        R{"negative"}=? [F C<={steps}]
        二者相减
        但 R=? 不能用在 nondeterministic models
        Rmin=? 和 Rmax=? 不能用 {"positive"} 和 {"negative"} 标记
        :param node:
        :return:
        """
        return f'state={node.state.tag} : {node.state.reward if node.state.reward >= 0 else 0};'

    def declaration(self):
        d = f'''
mdp

global sched: [0..{len(self.props_dict['name'])}] init 0;
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
state : [0..{self.node_num}] init 0;

[] (action={self.special_action_id}) & (sched=1) -> 1 : (sched\'={0 if len(self.props_dict["name"]) == 0 else 2});
{t}
endmodule
'''
        return code

    def parse_policy(self):
        t = ""
        r = ""
        for tag in self.policy_graph.nodes:
            if tag == self.node_num - 1:
                continue
            t += self.policy_transition(self.policy_graph.nodes[tag], signal='update') + "\n"
            if tag != 0:
                r += self.policy_reward(self.policy_graph.nodes[tag]) + "\n"

        code = f'''

module Policy
action: [0..{self.env_graph.action_spliter.action_len + 1}];

{t}
endmodule

rewards
{r}
endrewards
'''
        return code

    def parse_property(self):
        state0 = self.policy_graph.nodes[0].state
        codes = []

        for i, prop in enumerate(self.props_dict['name']):
            if prop == 'episode_reward':
                continue
            property_init = ''
            if hasattr(state0, prop):
                property_init += f'{prop}: [0..1] init 0;\n'

                p = f'[] ({prop}=1) & (sched={i+1}) -> 1: ({prop}\'=1) & (sched\'={(i+2)%(len(self.props_dict["name"])+1)});\n'
                if len(self.props_dict['name']) > 0:
                    for tag in self.policy_graph.nodes:
                        if tag == 0 or tag == self.node_num - 1:
                            continue
                        prob = float(getattr(self.policy_graph.nodes[tag].state, prop))
                        if prob > 1e-4:
                            p += self.add_transition(f'(state={tag}) & (sched={i+1})', updates=[[f'1', f'({prop}\'=1) & (sched\'={(i+2)%(len(self.props_dict["name"])+1)})']]) + '\n'
                code = f'''
        
module Property_{prop}
{property_init}
{p}
[](sched={i+1}) -> (sched\'={(i+2)%(len(self.props_dict["name"])+1)});
endmodule
'''
                codes.append(code)

        return '\n'.join(codes)

    def parse(self, save=False, file_path='./code.prism') -> str:
        """
        转换成 prism 代码
        :param save: 是否保存
        :param file_path: 文件名
        :return: prism 代码
        """
        code = f'{self.declaration()}\n{self.parse_policy()}\n{self.parse_env()}\n{self.parse_property()}'
        if save:
            print('save code to', file_path)
            if not os.path.exists(os.path.dirname(file_path)):
                os.mkdir(os.path.dirname(file_path))
            try:
                with open(file_path, 'w') as f:
                    f.write(code)
            except FileNotFoundError:
                print('file not found')

        return code

    def gen_property_file(self, avg_step: int, save=False, file_path='./property.props') -> str:
        """
        生成 property 文件
        :param avg_step: 平均步数
        :param save: 是否保存
        :param file_path: 文件名
        :return: property 代码
        """
        state0 = self.policy_graph.nodes[0].state
        properties = [f'Rmin=? [C<={avg_step}]']
        for i, prop in enumerate(self.props_dict['name']):
            if hasattr(state0, prop):
                properties.append(f'Pmax=? [ F<={avg_step} {prop}=1]')

        code = '\n'.join(properties)
        if save:
            print('save props to', file_path)
            if not os.path.exists(os.path.dirname(file_path)):
                os.mkdir(os.path.dirname(file_path))
            try:
                with open(file_path, 'w') as f:
                    f.write(code)
            except FileNotFoundError:
                print('file not found')

        return code
